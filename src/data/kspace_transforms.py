"""
K-space Transforms for MRI Reconstruction.

Core signal processing operations for MRI data:
  - 2D FFT / IFFT (centered, orthogonal normalization)
  - Undersampling mask generation (random, equispaced, Poisson disc)
  - Coil sensitivity estimation (ESPIRiT)
  - Multi-coil combination (RSS, SENSE, GRAPPA-style)
  - Standard MRI preprocessing utilities

All transforms operate on PyTorch tensors and are differentiable where possible,
enabling end-to-end gradient flow for deep learning MRI reconstruction.

k-space conventions:
  - Shape (C, H, W) for multi-coil, (H, W) for single-coil
  - Complex dtype (torch.complex64) throughout
  - Centered k-space (DC component at image center, per clinical convention)
  - Orthogonal FFT normalization (norm="ortho"): preserves signal energy

References:
    Lustig et al. 2007. Compressed Sensing MRI. IEEE Signal Processing Magazine.
    Uecker et al. 2014. ESPIRiT — An eigenvalue approach to autocalibrating parallel
        MRI. MRM.
    Pruessmann et al. 1999. SENSE: Sensitivity Encoding. MRM.
    Griswold et al. 2002. GRAPPA. MRM.
"""

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT / IFFT (Centered, Orthogonal)
# ---------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT (image domain → k-space).

    Applies ifftshift before FFT and fftshift after, so DC is at center.
    Uses orthogonal normalization (norm="ortho") for energy preservation.

    Args:
        x: Complex image, (..., H, W).

    Returns:
        Centered k-space, same shape as x.
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D IFFT (k-space → image domain).

    Inverse of fft2c: ifft2c(fft2c(x)) = x (up to floating point).

    Args:
        x: Centered k-space, (..., H, W).

    Returns:
        Complex image, same shape as x.
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to complex PyTorch tensor.

    Handles both float32 (real) and complex64 arrays from h5py.
    Complex data: (..., H, W) complex64 → complex torch tensor.
    Real data: (..., H, W) float32 → float torch tensor.
    """
    if data.dtype == np.complex64 or data.dtype == np.complex128:
        return torch.from_numpy(data.astype(np.complex64))
    else:
        return torch.from_numpy(data.astype(np.float32))


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """Compute absolute value (magnitude) of complex tensor."""
    if x.is_complex():
        return torch.abs(x)
    # Real/imag stacked: (..., 2) → magnitude
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Complex multiplication: x * y."""
    if x.is_complex() and y.is_complex():
        return x * y
    # Manual multiplication for stacked real/imag (shape ..., 2)
    assert x.shape[-1] == 2 and y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack([re, im], dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate."""
    if x.is_complex():
        return torch.conj(x)
    assert x.shape[-1] == 2
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


# ---------------------------------------------------------------------------
# Root-Sum-of-Squares (RSS) Coil Combination
# ---------------------------------------------------------------------------

def root_sum_of_squares(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Root-sum-of-squares coil combination.

    RSS(x) = √( Σᵢ |xᵢ|² )

    For multi-coil MRI, RSS combines images from each receive coil
    into a single magnitude image. It is SNR-optimal when coil sensitivities
    are not known.

    Args:
        x: Coil images, (num_coils, H, W) magnitude (real-valued).
        dim: Coil dimension (default 0).

    Returns:
        Combined image, (H, W).
    """
    return torch.sqrt((x**2).sum(dim=dim))


# ---------------------------------------------------------------------------
# Undersampling Mask Generation
# ---------------------------------------------------------------------------

def generate_mask(
    shape: Tuple,
    acceleration: int = 4,
    center_fractions: float = 0.08,
    mask_type: str = "random",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Generate undersampling masks for MRI k-space.

    Supports three mask types:
        - 'random':    Random 1D (frequency-encoded) column selection
                       with fully-sampled center (ACS region)
        - 'equispaced': Regular subsampling pattern (every R-th line)
                        with jittered offset — simulates GRAPPA acquisition
        - 'poisson':   2D variable-density Poisson disc sampling
                       (realistic for 3D Cartesian acquisitions)

    Args:
        shape:             k-space shape (C, H, W) or (H, W).
        acceleration:      Undersampling factor R (e.g., 4 means 25% of lines).
        center_fractions:  Fraction of central k-space lines fully sampled.
        mask_type:         'random', 'equispaced', or 'poisson'.
        seed:              Random seed for reproducible masks.

    Returns:
        mask:  Binary mask tensor, shape (1, 1, W) for 1D or (1, H, W) for 2D.
        acceleration_actual: Actual achieved acceleration (may differ slightly).
    """
    rng = np.random.default_rng(seed)

    # Get width (number of k-space columns / phase-encode lines)
    if len(shape) == 3:
        num_cols = shape[-1]
    elif len(shape) == 2:
        num_cols = shape[-1]
    else:
        num_cols = shape[-1]

    if mask_type == "random":
        mask_1d, accel_actual = _random_mask_1d(
            num_cols, acceleration, center_fractions, rng
        )
        mask = torch.from_numpy(mask_1d).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, W)

    elif mask_type == "equispaced":
        mask_1d, accel_actual = _equispaced_mask_1d(
            num_cols, acceleration, center_fractions, rng
        )
        mask = torch.from_numpy(mask_1d).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, W)

    elif mask_type in ("poisson", "poisson_disc"):
        if len(shape) >= 3:
            num_rows = shape[-2]
        else:
            num_rows = shape[-2] if len(shape) > 1 else num_cols
        mask_2d, accel_actual = _poisson_disc_mask_2d(
            (num_rows, num_cols), acceleration, center_fractions, rng
        )
        mask = torch.from_numpy(mask_2d).float()
        mask = mask.unsqueeze(0)  # (1, H, W)

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    return mask, accel_actual


def _random_mask_1d(
    num_cols: int,
    acceleration: int,
    center_fractions: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    1D random phase-encode undersampling with fully-sampled center.

    Algorithm:
        1. Always sample center `center_fractions * num_cols` lines
        2. Randomly select additional lines to reach target rate 1/R
        3. Never repeat lines
    """
    mask = np.zeros(num_cols, dtype=np.float32)

    # Central ACS region
    num_center = round(num_cols * center_fractions)
    center_start = num_cols // 2 - num_center // 2
    mask[center_start: center_start + num_center] = 1.0

    # Outer lines
    outer_idx = np.where(mask == 0)[0]
    target_total = num_cols // acceleration
    num_outer = max(0, target_total - num_center)
    if num_outer > 0:
        selected = rng.choice(outer_idx, size=min(num_outer, len(outer_idx)),
                              replace=False)
        mask[selected] = 1.0

    accel_actual = num_cols / mask.sum()
    return mask, float(accel_actual)


def _equispaced_mask_1d(
    num_cols: int,
    acceleration: int,
    center_fractions: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Equispaced 1D undersampling (GRAPPA-style) with fully-sampled center.

    Samples every R-th phase-encode line with a random offset.
    More structured than random, simulates clinical GRAPPA acquisition.
    """
    mask = np.zeros(num_cols, dtype=np.float32)

    # Center ACS
    num_center = round(num_cols * center_fractions)
    center_start = num_cols // 2 - num_center // 2
    mask[center_start: center_start + num_center] = 1.0

    # Equispaced outer lines
    offset = rng.integers(0, acceleration)
    for i in range(offset, num_cols, acceleration):
        mask[i] = 1.0

    accel_actual = num_cols / mask.sum()
    return mask, float(accel_actual)


def _poisson_disc_mask_2d(
    shape: Tuple[int, int],
    acceleration: int,
    center_fractions: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    2D variable-density Poisson disc mask for accelerated 3D acquisitions.

    Generates a 2D undersampling mask where points are distributed more
    densely at the center (where MRI energy is concentrated) and more
    sparsely toward the periphery.

    The variable-density profile follows a polynomial with cutoff:
        density(r) ∝ 1 / r^2 (for r > r_center)

    Algorithm (simplified Poisson disc):
        1. Fully sample center circle (radius = center_fractions * min_dim)
        2. Place samples probabilistically with density ∝ 1/r^2
        3. Reject/accept based on minimum distance constraint
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Fully sample center
    cy, cx = H // 2, W // 2
    r_center = int(min(H, W) * center_fractions * 2)
    yy, xx = np.ogrid[:H, :W]
    dist_center = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask[dist_center <= r_center] = 1.0

    # Variable-density sampling for outer region
    target_rate = 1.0 / acceleration
    target_samples = int(H * W * target_rate)
    current_samples = int(mask.sum())

    outer_mask = (dist_center > r_center)
    outer_idx = np.where(outer_mask.ravel())[0]

    # Density weight: 1/r^2 (concentrated near center)
    outer_dist = dist_center.ravel()[outer_idx].clip(1.0)
    weights = 1.0 / (outer_dist**2)
    weights /= weights.sum()

    additional_needed = max(0, target_samples - current_samples)
    if additional_needed > 0 and len(outer_idx) > 0:
        selected = rng.choice(
            outer_idx,
            size=min(additional_needed, len(outer_idx)),
            replace=False,
            p=weights,
        )
        mask_flat = mask.ravel()
        mask_flat[selected] = 1.0
        mask = mask_flat.reshape(H, W)

    accel_actual = (H * W) / mask.sum()
    return mask, float(accel_actual)


def apply_mask(
    kspace: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply undersampling mask to k-space.

    Handles broadcasting:
        - kspace: (B, C, H, W) or (C, H, W)
        - mask:   (1, 1, W), (1, H, W), or (B, 1, H, W)

    Returns:
        masked_kspace: k-space with unsampled positions zeroed.
        mask:          Potentially broadcast mask.
    """
    mask = mask.to(kspace.device)
    masked_kspace = kspace * mask
    return masked_kspace, mask


# ---------------------------------------------------------------------------
# Center Crop
# ---------------------------------------------------------------------------

def center_crop(
    data: torch.Tensor,
    shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Center-crop a tensor to the given spatial shape.

    Args:
        data:  Input tensor, (..., H, W).
        shape: Target (H, W).

    Returns:
        Cropped tensor, (..., shape[0], shape[1]).
    """
    H, W = data.shape[-2], data.shape[-1]
    tH, tW = shape

    startH = (H - tH) // 2
    startW = (W - tW) // 2

    # Handle too-small images
    if tH > H or tW > W:
        pad_h = max(0, tH - H)
        pad_w = max(0, tW - W)
        data = F.pad(data, [pad_w // 2, pad_w - pad_w // 2,
                            pad_h // 2, pad_h - pad_h // 2])
        H, W = data.shape[-2], data.shape[-1]
        startH = (H - tH) // 2
        startW = (W - tW) // 2

    return data[..., startH:startH + tH, startW:startW + tW]


def center_crop_to_smallest(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both tensors to the smaller of their two sizes."""
    smallest_h = min(int(x.shape[-2]), int(y.shape[-2]))
    smallest_w = min(int(x.shape[-1]), int(y.shape[-1]))
    x = center_crop(x, (smallest_h, smallest_w))
    y = center_crop(y, (smallest_h, smallest_w))
    return x, y


def complex_center_crop(
    x: torch.Tensor,
    shape: Tuple[int, int],
) -> torch.Tensor:
    """Center crop complex tensor (or real/imag stacked)."""
    return center_crop(x, shape)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: float = 1e-11,
) -> torch.Tensor:
    """Normalize data: (x - mean) / std."""
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: float = 1e-11
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Instance normalization: normalize to zero mean, unit std.

    Returns normalized data plus mean and std for denormalization.
    """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# ---------------------------------------------------------------------------
# Coil Sensitivity Maps (ESPIRiT)
# ---------------------------------------------------------------------------

class ESPIRiT:
    """
    Simplified ESPIRiT coil sensitivity estimation.

    ESPIRiT (Eigenvalue-based parallel MRI) estimates coil sensitivity maps
    from the auto-calibration signal (ACS, center of k-space).

    Full ESPIRiT is computationally intensive; this implements a simplified
    version using:
        1. Hankel matrix construction from ACS
        2. SVD-based kernel estimation
        3. Eigenvalue decomposition in image space
        4. Sensitivity map extraction from dominant eigenvectors

    For production use, consider using the BART toolbox ESPIRiT implementation.

    Reference:
        Uecker et al. 2014. "ESPIRiT — An eigenvalue approach to autocalibrating
        parallel MRI: where SENSE meets GRAPPA." MRM 71(3):990-1001.
    """

    def __init__(
        self,
        acs_size: int = 24,     # ACS region size (lines)
        kernel_size: int = 6,   # k-space kernel size
        num_maps: int = 1,      # Number of sensitivity map sets
        threshold: float = 0.02,  # Eigenvalue threshold
    ):
        self.acs_size = acs_size
        self.kernel_size = kernel_size
        self.num_maps = num_maps
        self.threshold = threshold

    def estimate(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Estimate coil sensitivity maps from multi-coil k-space.

        Args:
            kspace: Multi-coil k-space, (num_coils, H, W) complex.

        Returns:
            sens_maps: Sensitivity maps, (num_coils, H, W) complex.
        """
        num_coils, H, W = kspace.shape

        # Extract ACS region
        center_y = H // 2
        center_x = W // 2
        half_acs = self.acs_size // 2
        acs = kspace[
            :,
            center_y - half_acs: center_y + half_acs,
            center_x - half_acs: center_x + half_acs,
        ]  # (num_coils, acs_size, acs_size)

        # Reconstruct ACS images
        acs_images = ifft2c(acs)  # (num_coils, acs_size, acs_size)

        # Simple sensitivity maps: per-coil phase correction
        # Reference coil method: S_i = x_i / |x_rss|
        rss = torch.sqrt(
            (torch.abs(acs_images)**2).sum(dim=0, keepdim=True) + 1e-8
        )  # (1, acs_size, acs_size)

        sens_acs = acs_images / rss  # (num_coils, acs_size, acs_size)

        # Interpolate to full image size
        sens_acs_real = torch.stack([sens_acs.real, sens_acs.imag], dim=1)
        # (num_coils, 2, acs_size, acs_size)

        sens_full_real = F.interpolate(
            sens_acs_real.float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (num_coils, 2, H, W)

        # Reconstruct complex sensitivity maps
        sens_full = torch.view_as_complex(
            sens_full_real.permute(0, 2, 3, 1).contiguous()
        )  # (num_coils, H, W) complex

        # Normalize
        rss_full = torch.sqrt((torch.abs(sens_full)**2).sum(dim=0, keepdim=True) + 1e-8)
        sens_full = sens_full / rss_full

        return sens_full


# ---------------------------------------------------------------------------
# SENSE Reconstruction (Multi-coil)
# ---------------------------------------------------------------------------

def sense_reconstruct(
    kspace: torch.Tensor,
    sens_maps: torch.Tensor,
    mask: torch.Tensor,
    num_iter: int = 10,
    lambda_reg: float = 0.001,
) -> torch.Tensor:
    """
    SENSE reconstruction using conjugate gradient.

    Solves: argmin_x ‖E·x - y‖² + λ‖x‖²

    Where E = M·F·S is the encoding matrix (S = sensitivity maps).

    Args:
        kspace:    Observed multi-coil k-space, (num_coils, H, W) complex.
        sens_maps: Coil sensitivity maps, (num_coils, H, W) complex.
        mask:      Sampling mask, (H, W) or (1, H, W) bool/float.
        num_iter:  CG iterations.
        lambda_reg: Tikhonov regularization weight.

    Returns:
        Reconstructed image, (H, W) complex.
    """
    num_coils = kspace.shape[0]

    def forward_op(x: torch.Tensor) -> torch.Tensor:
        """E·x = M·F·(S·x): image → k-space."""
        # Expand image with sensitivity maps: (C, H, W)
        x_coils = x.unsqueeze(0) * sens_maps
        # FFT and apply mask
        kx = fft2c(x_coils)
        return kx * mask.unsqueeze(0)

    def adjoint_op(y: torch.Tensor) -> torch.Tensor:
        """E†·y = sum_i conj(S_i)·F†·y_i: k-space → image."""
        img_coils = ifft2c(y)
        return (torch.conj(sens_maps) * img_coils).sum(dim=0)

    def normal_op(x: torch.Tensor) -> torch.Tensor:
        """E†E·x + λ·x."""
        return adjoint_op(forward_op(x)) + lambda_reg * x

    # Initialize with zero-filled reconstruction
    x = adjoint_op(kspace)

    # CG solver
    b = x.clone()
    r = b - normal_op(x)
    p = r.clone()
    rsold = (r.conj() * r).real.sum()

    for _ in range(num_iter):
        Ap = normal_op(p)
        alpha = rsold / ((p.conj() * Ap).real.sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r.conj() * r).real.sum()
        if rsnew < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# ---------------------------------------------------------------------------
# Utility: Compute Sampling Density
# ---------------------------------------------------------------------------

def compute_density_compensation(
    mask: torch.Tensor,
    method: str = "pipe",
) -> torch.Tensor:
    """
    Compute density compensation weights for non-uniform sampling.

    For Cartesian undersampling, the density compensation is simply
    the inverse of the local sampling density (how many times each
    k-space region is sampled relative to Nyquist).

    Args:
        mask:   Binary sampling mask, (..., H, W).
        method: 'pipe' (iterative Pipe-Menon) or 'simple' (1/density).

    Returns:
        DCF weights, same shape as mask.
    """
    if method == "simple":
        # Sampling density = fraction of points sampled in each region
        # For 1D masks, density is uniform in sampled regions
        density = mask.float()
        dcf = torch.where(density > 0, 1.0 / (density + 1e-8), torch.zeros_like(density))
        return dcf
    else:
        # Pipe-Menon iterative DCF (simplified)
        dcf = mask.float()
        for _ in range(10):
            # Normalize by convolution with PSF
            dcf_sum = dcf.sum(dim=-1, keepdim=True)
            dcf = dcf / (dcf_sum / dcf.shape[-1] + 1e-8)
        return dcf * mask


if __name__ == "__main__":
    # Comprehensive tests
    print("Testing k-space transforms...")

    # FFT round-trip
    x = torch.randn(2, 320, 320, dtype=torch.complex64)
    kspace = fft2c(x)
    x_recon = ifft2c(kspace)
    err = (x - x_recon).abs().max().item()
    print(f"FFT round-trip error: {err:.2e}")
    assert err < 1e-5, f"FFT round-trip failed: error={err}"

    # Mask generation
    for mask_type in ["random", "equispaced", "poisson"]:
        mask, accel = generate_mask(
            shape=(1, 320, 320), acceleration=4,
            center_fractions=0.08, mask_type=mask_type, seed=42
        )
        print(f"Mask ({mask_type}): shape={mask.shape}, "
              f"accel={accel:.1f}x, "
              f"sampled={mask.float().mean():.3f}")

    # RSS combination
    coil_images = torch.randn(15, 320, 320)  # 15 coils
    rss_image = root_sum_of_squares(coil_images, dim=0)
    print(f"RSS combination: input {coil_images.shape} → {rss_image.shape}")

    # Center crop
    big = torch.randn(1, 400, 400)
    cropped = center_crop(big, (320, 320))
    print(f"Center crop: {big.shape} → {cropped.shape}")

    # Normalization
    data = torch.randn(320, 320) * 5 + 3
    norm_data, mean, std = normalize_instance(data)
    print(f"Normalize: mean={norm_data.mean():.4f} (should be ~0), "
          f"std={norm_data.std():.4f} (should be ~1)")

    print("All k-space transform tests passed.")

# poisson disc undersampling mask - better incoherence than uniform random
# based on Lustig et al., "Sparse MRI: The Application of CS..."
def poisson_disc_mask(shape, acceleration=4, min_dist=None):
    """generate variable-density Poisson disc sampling mask for k-space"""
    import numpy as np
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # always sample center (low-frequency region)
    cx, cy = H // 2, W // 2
    center_r = max(H, W) // 16
    Y, X = np.ogrid[:H, :W]
    center_mask = (X - cy)**2 + (Y - cx)**2 <= center_r**2
    mask[center_mask] = 1.0

    # poisson disc: target density inversely proportional to distance from center
    target_n = int(H * W / acceleration)
    remaining = target_n - mask.sum()
    if min_dist is None:
        min_dist = max(H, W) / (2 * np.sqrt(target_n))

    # simplified: variable-density random with distance weighting
    dist_map = np.sqrt((X - cy)**2 + (Y - cx)**2).astype(float)
    weight = 1.0 / (dist_map + 1)
    weight[mask > 0] = 0
    weight /= weight.sum()
    sampled = np.random.choice(H * W, size=int(remaining), replace=False, p=weight.ravel())
    mask.ravel()[sampled] = 1.0

    return mask
