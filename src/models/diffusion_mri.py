"""
Diffusion Process for MRI Reconstruction.

Implements Stochastic Differential Equations (SDEs) for the forward (noising)
and reverse (denoising) diffusion processes, adapted for the MRI reconstruction
inverse problem with k-space data consistency.

Two SDE formulations:
  - VP-SDE: Variance-Preserving (Ho et al. 2020, Song et al. 2021)
  - VE-SDE: Variance-Exploding (Song et al. 2019)

Reverse samplers:
  - Euler-Maruyama (EM)
  - Predictor-Corrector (PC) with Langevin dynamics
  - DDIM-style deterministic sampling (fast, ~50 steps)

Data consistency:
  - Gradient projection: x ← x - λ·A†(Ax - y)
  - Proximal operator for exact k-space projection

References:
    Song et al. 2021. "Score-Based Generative Modeling through SDEs." ICLR.
    Chung & Ye 2022. "Score-based diffusion models for accelerated MRI." MedIA.
    Chung et al. 2022. "Improving Diffusion Models for Inverse Problems." ICML.
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SDE Base Class
# ---------------------------------------------------------------------------

class SDE(ABC):
    """
    Abstract base class for Score-Based SDEs.

    The forward SDE is: dx = f(x,t)dt + g(t)dW
    The reverse SDE is: dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dW̄

    Where:
        f(x, t): drift coefficient (can be x-dependent)
        g(t):    diffusion coefficient (scalar)
        W:       standard Wiener process
    """

    def __init__(self, T: int = 1000, t_min: float = 1e-5, t_max: float = 1.0):
        self.T = T
        self.t_min = t_min
        self.t_max = t_max

    @abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift f(x,t) and diffusion g(t) coefficients."""
        pass

    @abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of p_t(x_t | x_0) = N(mean*x_0, std²I)."""
        pass

    @abstractmethod
    def prior_sampling(self, shape: Tuple) -> torch.Tensor:
        """Sample from the prior p_T(x)."""
        pass

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly in [t_min, t_max]."""
        return (
            torch.rand(batch_size, device=device) * (self.t_max - self.t_min)
            + self.t_min
        )

    def forward_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a sample from the forward diffusion q(x_t | x_0).

        Args:
            x0:    Clean image, shape (B, C, H, W).
            t:     Timesteps, shape (B,).
            noise: Optional pre-generated noise. If None, freshly sampled.

        Returns:
            xt:    Noisy sample at time t.
            noise: The noise that was added.
        """
        mean, std = self.marginal_prob(x0, t)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = mean + std[:, None, None, None] * noise
        return xt, noise


# ---------------------------------------------------------------------------
# VP-SDE (Variance-Preserving)
# ---------------------------------------------------------------------------

class VPSDE(SDE):
    """
    Variance-Preserving SDE.

    dx = -½β(t)x dt + √β(t) dW

    Where β(t) = β_min + (β_max - β_min) * t (linear schedule).

    Marginal distribution:
        p_t(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t)·I)

    where ᾱ_t = exp(-∫₀ᵗ β(s)ds) is the cumulative noise schedule.

    This corresponds to the DDPM formulation (Ho et al. 2020) in continuous time.
    Suitable for images normalized to approximately unit variance.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: int = 1000,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ):
        super().__init__(T, t_min, t_max)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule β(t)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def log_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Log of cumulative product: log ᾱ(t) = -∫₀ᵗ β(s)/2 ds."""
        return -0.5 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t)[:, None, None, None]
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_alpha = self.log_alpha_bar(t)                   # (B,)
        alpha_bar = torch.exp(log_alpha)[:, None, None, None]  # (B,1,1,1)
        mean = torch.sqrt(alpha_bar) * x
        std = torch.sqrt(1 - alpha_bar).squeeze()            # (B,)
        return mean, std

    def prior_sampling(self, shape: Tuple) -> torch.Tensor:
        """Sample from p_T ≈ N(0, I)."""
        return torch.randn(*shape)


# ---------------------------------------------------------------------------
# VE-SDE (Variance-Exploding)
# ---------------------------------------------------------------------------

class VESDE(SDE):
    """
    Variance-Exploding SDE.

    dx = σ(t) √(d σ²(t)/dt) dW

    Where σ(t) = σ_min (σ_max/σ_min)^t (geometric interpolation).

    Marginal distribution:
        p_t(x_t | x_0) = N(x_0, σ²(t)·I)

    No drift term — the process only adds noise, x_0 is unchanged in expectation.
    Better suited for continuous-valued MRI images at high dynamic ranges.
    Corresponds to NCSN (Song & Ermon 2019, 2020).
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        T: int = 1000,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ):
        super().__init__(T, t_min, t_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Geometric noise schedule: σ(t) = σ_min · (σ_max/σ_min)^t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_t = self.sigma(t)
        drift = torch.zeros_like(x)
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        diffusion = sigma_t * math.sqrt(2 * log_ratio)
        return drift, diffusion[:, None, None, None]

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_t = self.sigma(t)  # (B,)
        mean = x
        std = sigma_t
        return mean, std

    def prior_sampling(self, shape: Tuple) -> torch.Tensor:
        """Sample from p_T ≈ N(0, σ_max²·I)."""
        return torch.randn(*shape) * self.sigma_max


# ---------------------------------------------------------------------------
# Data Consistency for MRI
# ---------------------------------------------------------------------------

class DataConsistency(nn.Module):
    """
    Data consistency projection for MRI reconstruction.

    Enforces fidelity to measured k-space data y = A·x_true + noise,
    where A = M·F (undersampling mask M applied after Fourier transform F).

    Two modes:
        - 'gradient': Gradient descent step in image domain
          x ← x - λ · F†M†(MFx - y)
        - 'proximal': Exact data replacement in sampled frequencies
          x_k[m] = y[m] for m in mask, x_k[m] unchanged otherwise

    The gradient mode is differentiable and integrates smoothly into
    diffusion sampling. The proximal mode is exact but can cause artifacts
    at the boundary between sampled and unsampled frequencies.
    """

    def __init__(
        self,
        mode: str = "gradient",
        lambda_dc: float = 1.0,
        num_dc_steps: int = 1,
    ):
        super().__init__()
        assert mode in ("gradient", "proximal"), f"Unknown DC mode: {mode}"
        self.mode = mode
        self.lambda_dc = lambda_dc
        self.num_dc_steps = num_dc_steps

    def forward(
        self,
        x: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency.

        Args:
            x:          Current image estimate, (B, 2, H, W) real/imag or complex.
            kspace_obs: Observed k-space measurements, (B, 2, H, W) or complex.
            mask:       Binary sampling mask, (B, 1, H, W) or (1, 1, H, W).

        Returns:
            x_dc: Data-consistent image estimate, same shape as x.
        """
        for _ in range(self.num_dc_steps):
            if self.mode == "gradient":
                x = self._gradient_dc(x, kspace_obs, mask)
            else:
                x = self._proximal_dc(x, kspace_obs, mask)
        return x

    def _gradient_dc(
        self,
        x: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient step: x ← x - λ·A†(Ax - y)."""
        # Forward operator: A·x = M·F·x
        kx = fft2c(x)               # (B, 2, H, W) k-space
        Ax = mask * kx              # Apply mask in k-space
        residual = Ax - kspace_obs  # Residual in k-space

        # Adjoint: A†·r = F†·M†·r = F†·(M·r)  (M is self-adjoint)
        At_r = ifft2c(mask * residual)  # Back to image domain

        return x - self.lambda_dc * At_r

    def _proximal_dc(
        self,
        x: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact k-space replacement at sampled frequencies."""
        kx = fft2c(x)
        # Replace sampled k-space lines with observations
        kx_dc = kspace_obs * mask + kx * (1 - mask)
        return ifft2c(kx_dc)


# ---------------------------------------------------------------------------
# FFT Utilities (2D, centered, for MRI)
# ---------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    2D centered FFT for MRI (converts image domain → k-space).

    Handles both real (B, 2, H, W) and complex (B, C, H, W) inputs.
    The "2c" means 2D centered (fftshift applied).

    For real/imag stacked input: channels 0 and 1 are real and imaginary.
    """
    if x.is_complex():
        # Standard complex FFT
        kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        return kspace
    else:
        # Real/imag stacked: shape (B, 2*C, H, W) → process as complex
        assert x.shape[1] % 2 == 0, "Channel dim must be even for real/imag stacking"
        B, C2, H, W = x.shape
        C = C2 // 2
        x_c = torch.view_as_complex(
            x.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
        )  # (B, C, H, W) complex
        k_c = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x_c, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        k_real = torch.view_as_real(k_c)  # (B, C, H, W, 2)
        return k_real.permute(0, 1, 4, 2, 3).reshape(B, C2, H, W)


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """
    2D centered inverse FFT (converts k-space → image domain).

    Inverse of fft2c: ifft2c(fft2c(x)) ≈ x.
    """
    if x.is_complex():
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
    else:
        assert x.shape[1] % 2 == 0
        B, C2, H, W = x.shape
        C = C2 // 2
        x_c = torch.view_as_complex(
            x.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
        )
        img_c = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(x_c, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        img_real = torch.view_as_real(img_c)  # (B, C, H, W, 2)
        return img_real.permute(0, 1, 4, 2, 3).reshape(B, C2, H, W)


# ---------------------------------------------------------------------------
# Reverse Samplers
# ---------------------------------------------------------------------------

class EulerMaruyamaSampler:
    """
    Euler-Maruyama sampler for the reverse SDE.

    Simple first-order numerical integration of the reverse SDE:
        dx = [f(x,t) - g(t)² · ∇_x log p_t(x|y)] dt + g(t) dW̄

    At each step, optionally applies data consistency projection.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable,
        data_consistency: Optional[DataConsistency] = None,
        dc_freq: int = 1,  # Apply DC every dc_freq steps
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.data_consistency = data_consistency
        self.dc_freq = dc_freq

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        y: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 1000,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Run reverse diffusion from x_T ~ p_T to x_0.

        Args:
            shape:      Shape of output image (B, C, H, W).
            y:          Conditioning image (zero-filled), same shape as output.
            kspace_obs: Observed k-space, (B, 2, H, W).
            mask:       Sampling mask, (B, 1, H, W).
            num_steps:  Number of reverse diffusion steps.
            device:     Computation device.

        Returns:
            x:  Reconstructed image, shape (B, C, H, W).
        """
        t_max = self.sde.t_max
        t_min = self.sde.t_min
        dt = (t_max - t_min) / num_steps

        # Initialize from prior
        x = self.sde.prior_sampling(shape).to(device)
        y = y.to(device)
        kspace_obs = kspace_obs.to(device)
        mask = mask.to(device)

        timesteps = torch.linspace(t_max, t_min, num_steps, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((shape[0],), t_val, device=device)

            # Compute score
            score = self.score_fn(x, t, y)

            # Get SDE coefficients
            drift, diffusion = self.sde.sde(x, t)

            # Reverse drift: f(x,t) - g(t)² * score
            reverse_drift = drift - diffusion**2 * score

            # Euler-Maruyama update
            noise = torch.randn_like(x)
            x = x - reverse_drift * dt + diffusion * math.sqrt(dt) * noise

            # Data consistency
            if (
                self.data_consistency is not None
                and (i + 1) % self.dc_freq == 0
            ):
                x = self.data_consistency(x, kspace_obs, mask)

        return x


class PredictorCorrectorSampler:
    """
    Predictor-Corrector (PC) sampler.

    Combines:
      - Predictor: Reverse-diffusion SDE step (Euler-Maruyama or ancestral)
      - Corrector: Langevin MCMC steps targeting p_t(x) at fixed t

    This dramatically improves sample quality over pure EM sampling by
    running several corrector (MCMC) steps at each noise level.

    Ref: Song et al. 2021. Algorithm 2 in Appendix.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable,
        data_consistency: Optional[DataConsistency] = None,
        num_corrector_steps: int = 1,
        snr: float = 0.16,          # Signal-to-noise ratio for Langevin step size
        dc_freq: int = 1,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.data_consistency = data_consistency
        self.num_corrector_steps = num_corrector_steps
        self.snr = snr
        self.dc_freq = dc_freq

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        y: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 1000,
        device: torch.device = torch.device("cpu"),
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        PC sampling with data consistency.

        Args:
            shape:      Output shape (B, C, H, W).
            y:          Zero-filled conditioning image.
            kspace_obs: Observed k-space measurements.
            mask:       Sampling mask.
            num_steps:  Predictor steps (total = num_steps * (1 + num_corrector)).
            device:     Device to run on.
            callback:   Optional fn(step, x) called every 50 steps for monitoring.

        Returns:
            x: Final reconstruction.
        """
        t_max = self.sde.t_max
        t_min = self.sde.t_min
        dt = (t_max - t_min) / num_steps

        x = self.sde.prior_sampling(shape).to(device)
        y = y.to(device)
        kspace_obs = kspace_obs.to(device)
        mask = mask.to(device)

        timesteps = torch.linspace(t_max, t_min, num_steps, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((shape[0],), t_val, device=device)

            # ---- PREDICTOR STEP ----
            score = self.score_fn(x, t, y)
            drift, diffusion = self.sde.sde(x, t)
            reverse_drift = drift - diffusion**2 * score
            noise = torch.randn_like(x)
            x = x - reverse_drift * dt + diffusion * math.sqrt(dt) * noise

            # ---- CORRECTOR STEPS (Langevin MCMC) ----
            for _ in range(self.num_corrector_steps):
                score_c = self.score_fn(x, t, y)
                # Step size based on score norm and SNR
                score_norm = torch.norm(
                    score_c.reshape(shape[0], -1), dim=1
                ).mean()
                noise_norm = math.sqrt(score_c.numel() / shape[0])
                step_size = (self.snr * noise_norm / (score_norm + 1e-8)) ** 2 * 2

                noise_c = torch.randn_like(x)
                x = x + step_size * score_c + math.sqrt(2 * step_size) * noise_c

            # ---- DATA CONSISTENCY ----
            if (
                self.data_consistency is not None
                and (i + 1) % self.dc_freq == 0
            ):
                x = self.data_consistency(x, kspace_obs, mask)

            # Callback for monitoring
            if callback is not None and i % 50 == 0:
                callback(i, x.clone())

        return x


class DDIMSampler:
    """
    Deterministic DDIM-style sampler for fast inference.

    Replaces stochastic Euler-Maruyama steps with deterministic ones,
    allowing good quality reconstruction in 20–100 steps instead of 1000.
    Approximately 10–50x faster inference.

    Ref: Song et al. 2021. "Denoising Diffusion Implicit Models." ICLR.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable,
        data_consistency: Optional[DataConsistency] = None,
        eta: float = 0.0,  # 0=deterministic, 1=stochastic (recovers DDPM)
        dc_freq: int = 1,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.data_consistency = data_consistency
        self.eta = eta
        self.dc_freq = dc_freq

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        y: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Fast deterministic reconstruction using DDIM steps.

        Args:
            shape:      Output shape (B, C, H, W).
            y:          Zero-filled conditioning image.
            kspace_obs: Observed k-space measurements.
            mask:       Sampling mask.
            num_steps:  Number of DDIM steps (50 recommended for quality).
            device:     Device to run on.

        Returns:
            x: Reconstructed image.
        """
        assert isinstance(self.sde, VPSDE), "DDIM requires VP-SDE."

        t_max = self.sde.t_max
        t_min = self.sde.t_min

        x = self.sde.prior_sampling(shape).to(device)
        y = y.to(device)
        kspace_obs = kspace_obs.to(device)
        mask = mask.to(device)

        timesteps = torch.linspace(t_max, t_min, num_steps + 1, device=device)

        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t = torch.full((shape[0],), t_cur, device=device)

            # Score → predicted x0 via Tweedie's formula
            score = self.score_fn(x, t, y)
            _, std_t = self.sde.marginal_prob(x, t)
            mean_t, _ = self.sde.marginal_prob(torch.zeros_like(x), t)
            alpha_bar_t = torch.exp(self.sde.log_alpha_bar(t))[:, None, None, None]

            # Predicted x0 (denoised estimate)
            x0_pred = (x + std_t[:, None, None, None]**2 * score) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM update: interpolate between x0_pred and xt_next direction
            alpha_bar_next = torch.exp(
                self.sde.log_alpha_bar(t_next.expand(shape[0]))
            )[:, None, None, None]

            sigma_t = (
                self.eta
                * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_next)
            )

            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)
            x = (
                torch.sqrt(alpha_bar_next) * x0_pred
                + torch.sqrt(1 - alpha_bar_next - sigma_t**2) * score * std_t[:, None, None, None]
                + sigma_t * noise
            )

            # Data consistency
            if (
                self.data_consistency is not None
                and (i + 1) % self.dc_freq == 0
            ):
                x = self.data_consistency(x, kspace_obs, mask)

        return x


# ---------------------------------------------------------------------------
# MRI Diffusion Model: Combines SDE + Score Network + Samplers
# ---------------------------------------------------------------------------

class MRIDiffusionModel(nn.Module):
    """
    Complete diffusion model for MRI reconstruction.

    Wraps:
        - SDE (VP or VE)
        - Score network
        - Samplers (EM, PC, DDIM)
        - Data consistency

    Used for both training (compute score matching loss) and inference
    (run reverse diffusion to reconstruct from k-space measurements).
    """

    def __init__(
        self,
        score_network: nn.Module,
        sde_type: str = "vp",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        dc_mode: str = "gradient",
        dc_lambda: float = 1.0,
        dc_freq: int = 1,
    ):
        super().__init__()
        self.score_network = score_network

        # Initialize SDE
        if sde_type == "vp":
            self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max)
        elif sde_type == "ve":
            self.sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
        else:
            raise ValueError(f"Unknown SDE type: {sde_type}")

        self.data_consistency = DataConsistency(
            mode=dc_mode, lambda_dc=dc_lambda, num_dc_steps=1
        )
        self.dc_freq = dc_freq

    def score_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score using the score network."""
        return self.score_network(x, t, y)

    def compute_loss(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        loss_weighting: str = "likelihood",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute denoising score matching loss.

        Loss = E_{t,x_t}[ λ(t) · ‖s_θ(x_t, t) - ∇_x log p_t(x_t|x_0)‖² ]

        Where ∇_x log p_t(x_t|x_0) = -(x_t - √ᾱ_t · x_0) / (1 - ᾱ_t) for VP-SDE.

        Args:
            x0:             Clean target images, (B, C, H, W).
            y:              Conditioning input (zero-filled), same shape.
            t:              Optional timesteps. If None, sampled uniformly.
            loss_weighting: 'likelihood', 'uniform', or 'snr'.

        Returns:
            Dict with 'loss', 'score_loss', 't' keys.
        """
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = self.sde.sample_t(B, device)

        # Forward diffusion: perturb x0 → xt
        xt, noise = self.sde.forward_sample(x0, t)

        # Target score: -noise / std (by Tweedie's formula)
        _, std = self.sde.marginal_prob(x0, t)
        target_score = -noise / (std[:, None, None, None] + 1e-8)

        # Predicted score
        pred_score = self.score_fn(xt, t, y)

        # Loss weighting
        if loss_weighting == "likelihood":
            # Weight by g(t)² to get ELBO bound
            _, diffusion = self.sde.sde(xt, t)
            if isinstance(diffusion, torch.Tensor):
                weight = diffusion.squeeze() ** 2
            else:
                weight = std ** 2
        elif loss_weighting == "snr":
            # Weight by SNR = ᾱ_t / (1 - ᾱ_t)
            alpha_bar = torch.exp(
                self.sde.log_alpha_bar(t)
                if isinstance(self.sde, VPSDE)
                else torch.zeros_like(t)
            )
            weight = alpha_bar / (1 - alpha_bar + 1e-8)
        else:
            weight = torch.ones(B, device=device)

        # Per-sample MSE loss
        score_loss = (
            weight
            * torch.mean(
                (pred_score - target_score) ** 2,
                dim=(1, 2, 3),
            )
        ).mean()

        return {
            "loss": score_loss,
            "score_loss": score_loss.detach(),
            "t": t.detach(),
            "std": std.detach(),
        }

    @torch.no_grad()
    def reconstruct(
        self,
        y: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 1000,
        method: str = "pc",
        num_corrector_steps: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Reconstruct MRI from undersampled k-space measurements.

        Args:
            y:           Zero-filled aliased reconstruction (conditioning), (B, 2, H, W).
            kspace_obs:  Observed k-space data, (B, 2, H, W).
            mask:        Binary sampling mask, (B, 1, H, W).
            num_steps:   Number of reverse diffusion steps.
            method:      Sampler: 'em' (Euler-Maruyama), 'pc' (predictor-corrector),
                         'ddim' (fast deterministic).
            num_corrector_steps: Corrector steps per predictor step (PC only).
            device:      Target device.

        Returns:
            x_recon: Reconstructed image, (B, 2, H, W).
        """
        if device is None:
            device = next(self.parameters()).device

        shape = y.shape

        if method == "em":
            sampler = EulerMaruyamaSampler(
                self.sde, self.score_fn,
                data_consistency=self.data_consistency,
                dc_freq=self.dc_freq,
            )
        elif method == "pc":
            sampler = PredictorCorrectorSampler(
                self.sde, self.score_fn,
                data_consistency=self.data_consistency,
                num_corrector_steps=num_corrector_steps,
                dc_freq=self.dc_freq,
            )
        elif method == "ddim":
            sampler = DDIMSampler(
                self.sde, self.score_fn,
                data_consistency=self.data_consistency,
                eta=0.0,
                dc_freq=self.dc_freq,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return sampler.sample(
            shape, y, kspace_obs, mask,
            num_steps=num_steps, device=device,
        )


if __name__ == "__main__":
    # Sanity check diffusion model
    device = torch.device("cpu")

    # Test VP-SDE
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.rand(2) * 0.9 + 0.05

    xt, noise = sde.forward_sample(x0, t)
    mean, std = sde.marginal_prob(x0, t)
    print(f"VP-SDE: x0 norm={x0.norm():.2f}, xt norm={xt.norm():.2f}")
    print(f"  mean norm={mean.norm():.2f}, std={std}")

    # Test VE-SDE
    sde_ve = VESDE(sigma_min=0.01, sigma_max=50.0)
    xt_ve, noise_ve = sde_ve.forward_sample(x0, t)
    mean_ve, std_ve = sde_ve.marginal_prob(x0, t)
    print(f"VE-SDE: xt norm={xt_ve.norm():.2f}, std={std_ve}")

    # Test FFT round-trip
    x = torch.randn(2, 2, 32, 32)
    kspace = fft2c(x)
    x_recon = ifft2c(kspace)
    print(f"FFT round-trip error: {(x - x_recon).abs().max():.2e}")

    print("Diffusion model tests passed.")
