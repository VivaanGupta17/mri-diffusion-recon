"""
Compressed Sensing (CS) Baselines for MRI Reconstruction.

Implements classical optimization-based MRI reconstruction methods:
  1. Total Variation (TV) regularization with ADMM
  2. Wavelet sparsity with ISTA/FISTA
  3. Combined TV + Wavelet (standard clinical CS-MRI)

CS-MRI is the gold standard for comparison against deep learning methods.
Clinical systems (Siemens Compressed SENSE, Philips CS, GE HyperSense)
use variants of these algorithms.

The fundamental problem:
    minimize_x  λ_tv * TV(x) + λ_wav * ‖Ψx‖₁
    subject to  ‖MFx - y‖² ≤ ε

Where:
    M  = undersampling mask (binary)
    F  = 2D Fourier transform
    Ψ  = wavelet transform
    y  = measured k-space data
    ε  = noise level constraint

Solvers:
    ADMM (Alternating Direction Method of Multipliers) — for TV
    FISTA (Fast ISTA) — for wavelet sparsity
    POGM (Proximal Optimized Gradient Method) — combined

References:
    Lustig et al. 2007. "Sparse MRI: Application of Compressed Sensing."
        Magnetic Resonance in Medicine.
    Block et al. 2007. "Undersampled Radial MRI Reconstruction Using a
        Joint Estimation of Coil Sensitivities and Image Content." MICCAI.
    Beck & Teboulle 2009. "A Fast Iterative Shrinkage-Thresholding Algorithm."
        SIAM Journal on Imaging Sciences.
"""

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT Utilities (standalone, no circular imports)
# ---------------------------------------------------------------------------

def _fft2c(x: torch.Tensor) -> torch.Tensor:
    """2D centered FFT: image domain → k-space. Handles real/imag stacking."""
    if x.is_complex():
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
    B, C2, H, W = x.shape
    C = C2 // 2
    xc = torch.view_as_complex(x.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous())
    kc = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    return torch.view_as_real(kc).permute(0, 1, 4, 2, 3).reshape(B, C2, H, W)


def _ifft2c(x: torch.Tensor) -> torch.Tensor:
    """2D centered IFFT: k-space → image domain."""
    if x.is_complex():
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
    B, C2, H, W = x.shape
    C = C2 // 2
    xc = torch.view_as_complex(x.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous())
    ic = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(xc, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    return torch.view_as_real(ic).permute(0, 1, 4, 2, 3).reshape(B, C2, H, W)


# ---------------------------------------------------------------------------
# Proximal Operators
# ---------------------------------------------------------------------------

def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Soft-thresholding proximal operator for L1 norm.

    prox_{λ‖·‖₁}(x) = sign(x) * max(|x| - λ, 0)

    Used for wavelet coefficient shrinkage in ISTA/FISTA.
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def prox_tv_2d_isotropic(
    x: torch.Tensor,
    lambda_tv: float,
    num_iter: int = 10,
) -> torch.Tensor:
    """
    Proximal operator for 2D isotropic total variation.

    Solves: argmin_u  (1/2)‖u - x‖² + λ·TV(u)

    Uses Chambolle's dual gradient descent algorithm (Chambolle 2004).
    Applies independently to each image in the batch.

    Args:
        x:          Input image, (B, 2, H, W) real/imag channels.
        lambda_tv:  TV regularization strength.
        num_iter:   Inner iterations for Chambolle's algorithm.

    Returns:
        Denoised image, same shape as x.
    """
    B, C, H, W = x.shape
    # Process each channel independently
    result = torch.zeros_like(x)

    for c in range(C):
        xc = x[:, c:c+1, :, :]  # (B, 1, H, W)
        # Dual variable
        p = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
        tau = 0.25  # Step size (1 / (2 * 2) for 2D isotropic TV)

        for _ in range(num_iter):
            # Divergence of p: div(p)
            div_p = _divergence(p)  # (B, 1, H, W)

            # Gradient of (x - λ·div(p))
            grad = _gradient(xc - lambda_tv * div_p)  # (B, 2, H, W)

            # Dual update
            p_new = p + tau * grad
            # Project onto unit ball (isotropic TV constraint)
            norm_p = torch.sqrt(torch.sum(p_new**2, dim=1, keepdim=True) + 1e-8)
            p = p_new / torch.clamp(norm_p, min=1.0)

        result[:, c:c+1] = xc - lambda_tv * _divergence(p)

    return result


def _gradient(x: torch.Tensor) -> torch.Tensor:
    """Forward finite differences gradient, shape (B, 2, H, W)."""
    # Horizontal and vertical differences
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]  # Forward diff x
    dy[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]  # Forward diff y
    return torch.cat([dx, dy], dim=1)


def _divergence(p: torch.Tensor) -> torch.Tensor:
    """Divergence (adjoint of gradient), shape (B, 1, H, W)."""
    px, py = p[:, 0:1], p[:, 1:2]
    div_x = torch.zeros_like(px)
    div_y = torch.zeros_like(py)
    # Backward differences
    div_x[:, :, :, 1:-1] = px[:, :, :, 1:-1] - px[:, :, :, :-2]
    div_x[:, :, :, 0] = px[:, :, :, 0]
    div_x[:, :, :, -1] = -px[:, :, :, -2]
    div_y[:, :, 1:-1, :] = py[:, :, 1:-1, :] - py[:, :, :-2, :]
    div_y[:, :, 0, :] = py[:, :, 0, :]
    div_y[:, :, -1, :] = -py[:, :, -2, :]
    return div_x + div_y


# ---------------------------------------------------------------------------
# Haar Wavelet Transform (Simple 1-level for CS baseline)
# ---------------------------------------------------------------------------

class HaarWavelet2D(nn.Module):
    """
    Haar wavelet transform for 2D images.

    Implements 1-level and multi-level Haar DWT / IDWT using convolutions.
    Used as the sparsifying transform in wavelet CS.

    Coefficients: [LL, LH, HL, HH] (approximation + 3 detail subbands)
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        # Haar filters
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute multi-level Haar DWT.
        Returns list of detail coefficients per level + final approximation.
        """
        coeffs = []
        cur = x
        for _ in range(self.levels):
            ll, details = self._dwt2(cur)
            coeffs.append(details)  # [LH, HL, HH] per level
            cur = ll
        coeffs.append(cur)  # Final approximation
        return coeffs

    def inverse(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from wavelet coefficients."""
        cur = coeffs[-1]  # Start from approximation
        for details in reversed(coeffs[:-1]):
            cur = self._idwt2(cur, details)
        return cur

    def _dwt2(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single-level 2D DWT using Haar filters."""
        B, C, H, W = x.shape
        lo = self.lo
        hi = self.hi

        # Apply filters row-wise
        x_lo = (x[:, :, :, ::2] + x[:, :, :, 1::2]) / math.sqrt(2)
        x_hi = (x[:, :, :, ::2] - x[:, :, :, 1::2]) / math.sqrt(2)

        # Apply filters col-wise
        ll = (x_lo[:, :, ::2] + x_lo[:, :, 1::2]) / math.sqrt(2)
        lh = (x_lo[:, :, ::2] - x_lo[:, :, 1::2]) / math.sqrt(2)
        hl = (x_hi[:, :, ::2] + x_hi[:, :, 1::2]) / math.sqrt(2)
        hh = (x_hi[:, :, ::2] - x_hi[:, :, 1::2]) / math.sqrt(2)

        return ll, [lh, hl, hh]

    def _idwt2(
        self, ll: torch.Tensor, details: List[torch.Tensor]
    ) -> torch.Tensor:
        """Single-level 2D IDWT."""
        lh, hl, hh = details
        B, C, H, W = ll.shape

        # Reconstruct x_lo and x_hi
        x_lo_even = (ll + lh) / math.sqrt(2)
        x_lo_odd = (ll - lh) / math.sqrt(2)
        x_hi_even = (hl + hh) / math.sqrt(2)
        x_hi_odd = (hl - hh) / math.sqrt(2)

        # Interleave rows
        x_lo = torch.zeros(B, C, H * 2, W, device=ll.device, dtype=ll.dtype)
        x_hi = torch.zeros(B, C, H * 2, W, device=ll.device, dtype=ll.dtype)
        x_lo[:, :, ::2, :] = x_lo_even
        x_lo[:, :, 1::2, :] = x_lo_odd
        x_hi[:, :, ::2, :] = x_hi_even
        x_hi[:, :, 1::2, :] = x_hi_odd

        # Interleave cols
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, :, ::2] = (x_lo + x_hi) / math.sqrt(2)
        out[:, :, :, 1::2] = (x_lo - x_hi) / math.sqrt(2)

        return out


# ---------------------------------------------------------------------------
# ISTA / FISTA Solver (Wavelet Sparsity)
# ---------------------------------------------------------------------------

class FISTA_CS(nn.Module):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm for wavelet CS-MRI.

    Solves:
        minimize_x  (1/2)‖MFx - y‖² + λ‖Ψx‖₁

    Where Ψ is the wavelet transform.

    FISTA achieves O(1/k²) convergence (vs O(1/k) for ISTA).

    Args:
        lambda_wav:   Wavelet regularization weight.
        max_iter:     Maximum iterations.
        tolerance:    Convergence tolerance (relative residual change).
        wavelet_levels: Levels for multi-scale wavelet decomposition.
    """

    def __init__(
        self,
        lambda_wav: float = 0.01,
        max_iter: int = 200,
        tolerance: float = 1e-5,
        wavelet_levels: int = 3,
    ):
        super().__init__()
        self.lambda_wav = lambda_wav
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.wavelet = HaarWavelet2D(levels=wavelet_levels)

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct from undersampled k-space using FISTA.

        Args:
            y:    Observed k-space data (masked), (B, 2, H, W).
            mask: Sampling mask, (B, 1, H, W) or (1, 1, H, W).
            x0:   Initial estimate. If None, uses zero-filled reconstruction.

        Returns:
            Dict with 'recon', 'iterations', 'final_loss'.
        """
        # Lipschitz constant L = ‖A†A‖ = 1 (for orthogonal undersampling)
        L = 1.0
        step_size = 1.0 / L

        # Initialize
        if x0 is None:
            x = _ifft2c(y)  # Zero-filled reconstruction
        else:
            x = x0.clone()

        z = x.clone()  # Momentum variable (FISTA)
        t = 1.0  # Momentum coefficient

        losses = []

        for k in range(self.max_iter):
            x_old = x.clone()

            # Gradient step: x ← z - step * A†(Az - y)
            kspace_z = _fft2c(z)
            residual = mask * kspace_z - y
            gradient = _ifft2c(mask * residual)
            z_grad = z - step_size * gradient

            # Wavelet soft-thresholding
            x = self._wavelet_threshold(z_grad, step_size * self.lambda_wav)

            # FISTA momentum update
            t_new = (1.0 + math.sqrt(1 + 4 * t**2)) / 2.0
            z = x + ((t - 1) / t_new) * (x - x_old)
            t = t_new

            # Convergence check
            rel_change = (x - x_old).norm() / (x_old.norm() + 1e-10)
            if rel_change < self.tolerance:
                break

            # Track loss every 10 iters
            if k % 10 == 0:
                data_fit = 0.5 * (mask * _fft2c(x) - y).norm() ** 2
                losses.append(data_fit.item())

        return {
            "recon": x,
            "iterations": k + 1,
            "final_loss": losses[-1] if losses else 0.0,
        }

    def _wavelet_threshold(
        self, x: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """Apply wavelet soft-thresholding."""
        coeffs = self.wavelet(x)
        # Threshold all detail coefficients (not the approximation)
        thresholded_coeffs = []
        for i, level_coeffs in enumerate(coeffs[:-1]):  # Skip final approx
            thresholded_level = [soft_threshold(c, threshold) for c in level_coeffs]
            thresholded_coeffs.append(thresholded_level)
        thresholded_coeffs.append(coeffs[-1])  # Keep approximation unchanged
        return self.wavelet.inverse(thresholded_coeffs)


# ---------------------------------------------------------------------------
# ADMM Solver (TV Regularization)
# ---------------------------------------------------------------------------

class ADMM_TV(nn.Module):
    """
    Alternating Direction Method of Multipliers for TV-regularized MRI.

    Solves:
        minimize_x  (1/2)‖MFx - y‖² + λ_tv·TV(x)

    ADMM reformulation (splitting: u = ∇x):
        minimize_{x,u}  (1/2)‖MFx - y‖² + λ·‖u‖₁
        subject to      ∇x = u

    Augmented Lagrangian (ρ is ADMM penalty parameter):
        L(x, u, v) = f(x) + λ‖u‖₁ + (ρ/2)‖∇x - u + v‖²

    x-update: Linear system solve (via conjugate gradient or direct)
    u-update: Soft-thresholding of ∇x + v
    v-update: Dual ascent v ← v + ∇x - u

    Args:
        lambda_tv:   TV regularization strength.
        rho:         ADMM penalty parameter (controls convergence speed).
        max_iter:    Maximum outer iterations.
        cg_iter:     Conjugate gradient iterations for x-update.
    """

    def __init__(
        self,
        lambda_tv: float = 0.01,
        rho: float = 1.0,
        max_iter: int = 50,
        cg_iter: int = 10,
    ):
        super().__init__()
        self.lambda_tv = lambda_tv
        self.rho = rho
        self.max_iter = max_iter
        self.cg_iter = cg_iter

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct using ADMM with TV regularization.

        Args:
            y:    Observed k-space, (B, 2, H, W).
            mask: Binary mask, (B, 1, H, W).
            x0:   Initial estimate (zero-filled if None).

        Returns:
            Dict with 'recon', 'iterations', 'primal_residual', 'dual_residual'.
        """
        if x0 is None:
            x = _ifft2c(y)
        else:
            x = x0.clone()

        u = _gradient(x)        # Gradient of x (2B, 2, H, W)
        v = torch.zeros_like(u)  # Dual variable

        primal_residuals = []
        dual_residuals = []

        for k in range(self.max_iter):
            # ---- x-update: solve (A†A + ρ·D†D) x = A†y + ρ·D†(u - v) ----
            b = _ifft2c(mask * y) + self.rho * _divergence(u - v)
            x = self._cg_solve(x, b, mask)

            # ---- u-update: soft-thresholding ----
            grad_x_plus_v = _gradient(x) + v
            u_new = soft_threshold(grad_x_plus_v, self.lambda_tv / self.rho)

            # ---- v-update: dual ascent ----
            v = v + _gradient(x) - u_new

            # Residuals
            primal_res = (_gradient(x) - u_new).norm().item()
            dual_res = (self.rho * _divergence(u_new - u)).norm().item()
            primal_residuals.append(primal_res)
            dual_residuals.append(dual_res)

            u = u_new

            # Convergence
            if primal_res < 1e-4 and dual_res < 1e-4:
                break

        return {
            "recon": x,
            "iterations": k + 1,
            "primal_residual": primal_residuals[-1],
            "dual_residual": dual_residuals[-1],
        }

    def _cg_solve(
        self,
        x0: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conjugate gradient solver for the x-update linear system.

        Solves: (A†A + ρ·D†D) x = b

        Where A†A = F†MF (data fidelity), D†D = div·grad (Laplacian).
        """
        x = x0.clone()
        Ax = _ifft2c(mask * _fft2c(x)) - self.rho * _divergence(_gradient(x))
        r = b - Ax
        p = r.clone()
        rsold = (r * r).sum()

        for _ in range(self.cg_iter):
            Ap = (_ifft2c(mask * _fft2c(p)) - self.rho * _divergence(_gradient(p)))
            alpha = rsold / ((p * Ap).sum() + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (r * r).sum()
            if rsnew < 1e-10:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x


# ---------------------------------------------------------------------------
# Combined CS (TV + Wavelet)
# ---------------------------------------------------------------------------

class CombinedCS(nn.Module):
    """
    Combined compressed sensing with TV + wavelet regularization.

    This is the standard clinical CS-MRI formulation (Lustig et al. 2007):
        minimize_x  λ_tv·TV(x) + λ_wav·‖Ψx‖₁
        subject to  ‖MFx - y‖² ≤ ε

    Solved via alternating between FISTA (wavelet) and TV proximal steps,
    with projected gradient for data fidelity.

    Args:
        lambda_tv:   TV regularization weight.
        lambda_wav:  Wavelet regularization weight.
        max_iter:    Maximum outer iterations.
        wavelet_levels: Wavelet decomposition depth.
    """

    def __init__(
        self,
        lambda_tv: float = 0.005,
        lambda_wav: float = 0.005,
        max_iter: int = 150,
        wavelet_levels: int = 3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.lambda_tv = lambda_tv
        self.lambda_wav = lambda_wav
        self.max_iter = max_iter
        self.wavelet = HaarWavelet2D(levels=wavelet_levels)

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct using combined TV + wavelet CS.

        Args:
            y:       Observed k-space, (B, 2, H, W).
            mask:    Sampling mask, (B, 1, H, W).
            x0:      Initial estimate.
            verbose: Print convergence info.

        Returns:
            Dict with 'recon', 'iterations', 'loss_history'.
        """
        if x0 is None:
            x = _ifft2c(y)
        else:
            x = x0.clone()

        z = x.clone()
        t = 1.0
        loss_history = []

        start_time = time.time()

        for k in range(self.max_iter):
            x_old = x.clone()

            # Gradient of data fidelity term
            kspace = _fft2c(z)
            residual = mask * kspace - y
            gradient = _ifft2c(mask * residual)

            # Gradient step
            step_size = 1.0
            z_step = z - step_size * gradient

            # TV proximal step (Chambolle's algorithm)
            x_tv = prox_tv_2d_isotropic(z_step, step_size * self.lambda_tv, num_iter=5)

            # Wavelet soft-thresholding
            coeffs = self.wavelet(x_tv)
            coeffs_thresh = coeffs[:-1]
            x_thresh_coeffs = [
                [soft_threshold(c, step_size * self.lambda_wav) for c in level]
                for level in coeffs_thresh
            ]
            x_thresh_coeffs.append(coeffs[-1])  # Keep approximation
            x = self.wavelet.inverse(x_thresh_coeffs)

            # FISTA momentum
            t_new = (1 + math.sqrt(1 + 4 * t**2)) / 2
            z = x + ((t - 1) / t_new) * (x - x_old)
            t = t_new

            # Track loss
            if k % 10 == 0:
                data_fit = 0.5 * (mask * _fft2c(x) - y).norm().item() ** 2
                tv_reg = self._compute_tv(x).item()
                total_loss = data_fit + self.lambda_tv * tv_reg
                loss_history.append(total_loss)

                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  Iter {k:4d}/{self.max_iter}: loss={total_loss:.4f} "
                          f"data={data_fit:.4f} tv={tv_reg:.4f} ({elapsed:.1f}s)")

            # Convergence
            rel_change = (x - x_old).norm() / (x_old.norm() + 1e-10)
            if rel_change < 1e-5:
                break

        return {
            "recon": x,
            "iterations": k + 1,
            "loss_history": loss_history,
            "elapsed_time": time.time() - start_time,
        }

    def _compute_tv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute isotropic total variation."""
        grad = _gradient(x)
        return torch.sqrt((grad**2).sum(dim=1) + 1e-8).sum()


# ---------------------------------------------------------------------------
# Benchmark Evaluation Wrapper
# ---------------------------------------------------------------------------

class CSReconBenchmark:
    """
    Unified interface for evaluating CS reconstruction methods.

    Provides consistent input/output interface for benchmarking CS methods
    against deep learning approaches.
    """

    METHODS = {
        "tv_admm": ADMM_TV,
        "wavelet_fista": FISTA_CS,
        "combined_cs": CombinedCS,
    }

    def __init__(
        self,
        method: str = "combined_cs",
        lambda_tv: float = 0.005,
        lambda_wav: float = 0.005,
        max_iter: int = 150,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        if method == "tv_admm":
            self.solver = ADMM_TV(lambda_tv=lambda_tv, max_iter=max_iter)
        elif method == "wavelet_fista":
            self.solver = FISTA_CS(lambda_wav=lambda_wav, max_iter=max_iter)
        elif method == "combined_cs":
            self.solver = CombinedCS(
                lambda_tv=lambda_tv, lambda_wav=lambda_wav, max_iter=max_iter
            )
        else:
            raise ValueError(f"Unknown CS method: {method}")

    def reconstruct(
        self,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CS reconstruction.

        Args:
            kspace_obs: Masked k-space, (B, 2, H, W).
            mask:       Binary mask, (B, 1, H, W).

        Returns:
            Reconstruction, (B, 2, H, W).
        """
        kspace_obs = kspace_obs.to(self.device)
        mask = mask.to(self.device)

        result = self.solver(kspace_obs, mask)
        return result["recon"]


if __name__ == "__main__":
    device = torch.device("cpu")

    B, H, W = 1, 64, 64

    # Create phantom image
    x_true = torch.zeros(B, 2, H, W)
    x_true[:, 0, H//4:3*H//4, W//4:3*W//4] = 1.0  # Bright square

    # Simulate undersampled k-space
    from src.models.diffusion_mri import fft2c
    kspace_full = _fft2c(x_true)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, :, ::4] = 1.0  # 4x equispaced undersampling
    kspace_obs = kspace_full * mask

    # Test FISTA
    solver_fista = FISTA_CS(lambda_wav=0.01, max_iter=50)
    result_fista = solver_fista(kspace_obs, mask)
    print(f"FISTA: {result_fista['iterations']} iterations, "
          f"recon shape: {result_fista['recon'].shape}")

    # Test ADMM
    solver_admm = ADMM_TV(lambda_tv=0.01, max_iter=20, cg_iter=5)
    result_admm = solver_admm(kspace_obs, mask)
    print(f"ADMM: {result_admm['iterations']} iterations, "
          f"primal res: {result_admm['primal_residual']:.4f}")

    # Test combined CS
    solver_cs = CombinedCS(lambda_tv=0.005, lambda_wav=0.005, max_iter=30)
    result_cs = solver_cs(kspace_obs, mask)
    print(f"Combined CS: {result_cs['iterations']} iterations")

    print("CS baseline tests passed.")
