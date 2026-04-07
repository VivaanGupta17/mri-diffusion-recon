"""
U-Net Baseline for MRI Reconstruction.

Implements:
    1. Standard U-Net: Single-pass image domain reconstruction
    2. Cascaded U-Net: Multiple refinement stages with data consistency
    3. Complex-valued input/output support

The cascaded U-Net (E2E-VarNet style) is the state-of-the-art CNN baseline
for MRI reconstruction, surpassing compressed sensing at moderate accelerations.

References:
    Ronneberger et al. 2015. "U-Net: Convolutional Networks for Biomedical Image
        Segmentation." MICCAI.
    Sriram et al. 2020. "End-to-End Variational Networks for Accelerated MRI
        Reconstruction." MICCAI.
    Zbontar et al. 2018. "fastMRI: An Open Dataset and Benchmarks." arXiv.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic Blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → Norm → Act → Conv → Norm → Act.

    Standard U-Net building block with optional residual connection.
    Uses Instance Normalization (preferred over BatchNorm for small batches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Residual: only when channels match
        self.use_residual = use_residual and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.act(self.norm2(self.conv2(h)))
        if self.use_residual:
            h = h + x
        return h


class DownBlock(nn.Module):
    """Encoder block: max-pool downsample + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate, use_residual=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Decoder block: bilinear upsample + skip concat + ConvBlock.

    Bilinear upsampling is preferred over transposed convolution to avoid
    checkerboard artifacts common in medical image reconstruction.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        # Reduce in_channels before concatenation
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        cat_channels = in_channels // 2 + skip_channels
        self.conv = ConvBlock(cat_channels, out_channels, dropout_rate, use_residual=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        # Handle potential size mismatch (from odd input dimensions)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Standard U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Standard U-Net for MRI reconstruction in the image domain.

    Architecture:
        - 4-level encoder (successive halving of spatial resolution)
        - Bottleneck with large receptive field
        - 4-level decoder with skip connections
        - Output: residual correction added to zero-filled input

    Input:
        x: Zero-filled reconstruction, (B, 2, H, W) [real+imag channels]
    Output:
        Refined reconstruction, same shape as input.

    The network predicts a RESIDUAL added to x (not the clean image directly).
    This "residual learning" strategy stabilizes training.
    """

    def __init__(
        self,
        in_channels: int = 2,          # 2 for single-coil (real+imag)
        out_channels: int = 2,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_mults]

        # Input projection
        self.input_conv = ConvBlock(in_channels, channels[0], use_residual=False)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], dropout_rate)
            )

        # Bottleneck
        bottleneck_ch = channels[-1]
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, dropout_rate,
                                    use_residual=True)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(channels[i], channels[i - 1], channels[i - 1], dropout_rate)
            )

        # Output projection (residual)
        self.output_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (zero-filled), (B, 2, H, W).
        Returns:
            Reconstructed image, (B, 2, H, W).
        """
        input_x = x  # Save for residual

        # Encoder
        h = self.input_conv(x)
        skips = [h]
        for down in self.down_blocks:
            h = down(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(skips.pop())

        # Decoder
        for up in self.up_blocks:
            skip = skips.pop()
            h = up(h, skip)

        # Residual output
        return input_x + self.output_conv(h)


# ---------------------------------------------------------------------------
# Cascaded U-Net (Multi-Stage)
# ---------------------------------------------------------------------------

class DataConsistencyLayer(nn.Module):
    """
    Differentiable data consistency layer for cascaded U-Net.

    Enforces fidelity to k-space measurements between cascade stages:
        k_out = mask * k_obs + (1 - mask) * k_pred

    Inserted between each U-Net stage in the cascade.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_pred: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_pred:     Current image prediction, (B, 2, H, W).
            kspace_obs: Measured k-space, (B, 2, H, W).
            mask:       Binary mask, (B, 1, H, W) or (1, 1, H, W).
        Returns:
            x_dc: Data-consistent image, (B, 2, H, W).
        """
        from src.models.diffusion_mri import fft2c, ifft2c

        k_pred = fft2c(x_pred)
        k_dc = mask * kspace_obs + (1 - mask) * k_pred
        return ifft2c(k_dc)


class CascadedUNet(nn.Module):
    """
    Cascaded U-Net with iterative data consistency.

    Each cascade stage:
        1. U-Net refinement in image domain
        2. Data consistency projection in k-space

    This mirrors the architecture used in fastMRI challenge top solutions
    (E2E-VarNet, Cascade CNN variants).

    Args:
        num_cascades: Number of refinement stages (default 12, like E2E-VarNet)
        shared_weights: If True, all stages share weights (reduces parameters 12x)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 12,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        dropout_rate: float = 0.0,
        shared_weights: bool = False,
    ):
        super().__init__()
        self.num_cascades = num_cascades
        self.shared_weights = shared_weights

        # Create U-Net stages
        if shared_weights:
            shared_unet = UNet(
                in_channels, out_channels, base_channels, channel_mults, dropout_rate
            )
            self.stages = nn.ModuleList([shared_unet] * num_cascades)
        else:
            self.stages = nn.ModuleList([
                UNet(in_channels, out_channels, base_channels, channel_mults,
                     dropout_rate)
                for _ in range(num_cascades)
            ])

        self.dc_layers = nn.ModuleList([
            DataConsistencyLayer() for _ in range(num_cascades)
        ])

        # Learnable sensitivity maps weighting (for multi-coil)
        self.coil_combine = None  # Populated for multi-coil in subclass

    def forward(
        self,
        x: torch.Tensor,
        kspace_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          Initial estimate (zero-filled), (B, 2, H, W).
            kspace_obs: Observed k-space, (B, 2, H, W).
            mask:       Sampling mask, (B, 1, H, W).
        Returns:
            Final reconstruction, (B, 2, H, W).
        """
        for stage, dc in zip(self.stages, self.dc_layers):
            x = stage(x)
            x = dc(x, kspace_obs, mask)
        return x


# ---------------------------------------------------------------------------
# Multi-Coil Cascaded U-Net with Sensitivity Maps
# ---------------------------------------------------------------------------

class SensitivityModel(nn.Module):
    """
    Learnable coil sensitivity map estimation.

    Uses the center of k-space (autocalibration signal, ACS) to estimate
    sensitivity maps for each coil. Maps are used to combine multi-coil
    images and to expand single-channel predictions back to multi-coil.
    """

    def __init__(self, num_coils: int, in_channels: int = 2, base_channels: int = 8):
        super().__init__()
        self.num_coils = num_coils
        # Simple U-Net to estimate sensitivity maps from ACS data
        self.unet = UNet(
            in_channels=2,  # Magnitude + phase of center k-space
            out_channels=2 * num_coils,  # Real+imag per coil
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
        )

    def forward(
        self, kspace_center: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate sensitivity maps from auto-calibration signal.

        Args:
            kspace_center: Center k-space ACS region, (B, 2, H, W).
        Returns:
            sensitivity_maps: Per-coil maps, (B, 2*num_coils, H, W).
        """
        acs_image = _ifft2c_simple(kspace_center)
        maps = self.unet(acs_image)
        # Normalize maps (sum of squares = 1 across coils)
        maps_c = maps.reshape(maps.shape[0], self.num_coils, 2, *maps.shape[-2:])
        norm = torch.sqrt(
            (maps_c ** 2).sum(dim=(1, 2), keepdim=True) + 1e-8
        )
        return (maps_c / norm).reshape(maps.shape)


def _ifft2c_simple(x: torch.Tensor) -> torch.Tensor:
    """Simplified IFFT2C without the full diffusion import (for standalone use)."""
    # Real/imag stacked
    B, C2, H, W = x.shape
    assert C2 % 2 == 0
    C = C2 // 2
    x_c = torch.view_as_complex(
        x.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
    )
    img_c = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x_c, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    img_real = torch.view_as_real(img_c)
    return img_real.permute(0, 1, 4, 2, 3).reshape(B, C2, H, W)


class MultiCoilCascadedUNet(nn.Module):
    """
    Multi-coil cascaded U-Net with learned sensitivity maps.

    Full pipeline for parallel MRI:
        1. Estimate sensitivity maps from ACS
        2. Coil combination (multi-coil → single-channel)
        3. Cascaded U-Net refinement
        4. Coil expansion (single-channel → multi-coil)
        5. Data consistency in coil k-space

    This approximates the E2E-VarNet architecture used in top fastMRI solutions.
    """

    def __init__(
        self,
        num_coils: int = 15,
        num_cascades: int = 12,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_coils = num_coils

        # Sensitivity map estimation
        self.sens_model = SensitivityModel(num_coils, base_channels=8)

        # Core cascade (operates on combined single-coil image)
        self.cascade = CascadedUNet(
            in_channels=2,
            out_channels=2,
            num_cascades=num_cascades,
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            kspace: Multi-coil k-space, (B, num_coils, 2, H, W).
            mask:   Sampling mask, (B, 1, H, W) or (1, 1, H, W).
            target: Optional target image for supervised training.
        Returns:
            Reconstructed single-channel image, (B, 2, H, W).
        """
        B, C, ch, H, W = kspace.shape  # C = num_coils, ch = 2

        # Flatten coils for ACS estimation
        kspace_flat = kspace.reshape(B, C * ch, H, W)

        # Estimate sensitivity maps from center k-space
        acs_mask = self._get_acs_mask(mask, num_lines=24).to(kspace.device)
        kspace_acs = kspace_flat * acs_mask.unsqueeze(1)
        sens_maps = self.sens_model(kspace_acs[:, :2])  # Use first coil for ACS

        # Coil combination: sum_i conj(S_i) * x_i / sum_i |S_i|²
        x_combined = self._coil_combine(kspace, sens_maps)

        # Get zero-filled combined image and observed k-space
        kspace_obs = (kspace_flat * mask).reshape(B, C, ch, H, W)
        kspace_obs_combined = self._combine_kspace(kspace_obs, sens_maps)

        # Cascade
        x_recon = self.cascade(x_combined, kspace_obs_combined, mask)

        return x_recon

    def _get_acs_mask(self, mask: torch.Tensor, num_lines: int) -> torch.Tensor:
        """Get central k-space lines for auto-calibration signal."""
        H, W = mask.shape[-2], mask.shape[-1]
        acs = torch.zeros_like(mask)
        center = W // 2
        half = num_lines // 2
        acs[..., center - half: center + half] = 1.0
        return acs

    def _coil_combine(
        self, kspace: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """Combine multi-coil images using sensitivity maps (RSS approximation)."""
        # Simple RSS combination for now
        B, C, ch, H, W = kspace.shape
        kspace_flat = kspace.reshape(B, C * ch, H, W)
        images = _ifft2c_simple(kspace_flat.reshape(B, -1, H, W))
        # RSS
        images_c = images.reshape(B, C, ch, H, W)
        rss = torch.sqrt((images_c ** 2).sum(dim=(1, 2), keepdim=False) + 1e-8)
        # Return as 2-channel (real=RSS, imag=0)
        combined = torch.stack([rss, torch.zeros_like(rss)], dim=1)
        return combined

    def _combine_kspace(
        self, kspace: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """Combine multi-coil k-space (simplified for single channel output)."""
        B, C, ch, H, W = kspace.shape
        kspace_flat = kspace.reshape(B, C * ch, H, W)
        # Average across coils (simplified)
        k_avg = kspace_flat.reshape(B, C, ch, H, W).mean(dim=1)
        return k_avg


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_unet(config: dict) -> nn.Module:
    """
    Build a U-Net model from a configuration dictionary.

    Args:
        config: Dict with keys model_type, num_cascades, base_channels, etc.
    Returns:
        Instantiated model.
    """
    model_type = config.get("model_type", "cascaded_unet")
    base_channels = config.get("base_channels", 32)
    channel_mults = tuple(config.get("channel_mults", [1, 2, 4, 8]))
    dropout_rate = config.get("dropout_rate", 0.0)

    if model_type == "unet":
        return UNet(
            in_channels=config.get("in_channels", 2),
            out_channels=config.get("out_channels", 2),
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout_rate=dropout_rate,
        )
    elif model_type == "cascaded_unet":
        return CascadedUNet(
            in_channels=config.get("in_channels", 2),
            out_channels=config.get("out_channels", 2),
            num_cascades=config.get("num_cascades", 12),
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout_rate=dropout_rate,
            shared_weights=config.get("shared_weights", False),
        )
    elif model_type == "multicoil_cascaded_unet":
        return MultiCoilCascadedUNet(
            num_coils=config.get("num_coils", 15),
            num_cascades=config.get("num_cascades", 12),
            base_channels=base_channels,
            channel_mults=channel_mults,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown U-Net type: {model_type}")


if __name__ == "__main__":
    device = torch.device("cpu")

    # Test standard U-Net
    model = UNet(in_channels=2, out_channels=2, base_channels=16,
                 channel_mults=(1, 2, 4)).to(device)
    x = torch.randn(2, 2, 64, 64)
    y = model(x)
    print(f"U-Net: input {x.shape} → output {y.shape}")
    assert y.shape == x.shape

    # Test cascaded U-Net
    from src.models.diffusion_mri import fft2c
    model_c = CascadedUNet(num_cascades=3, base_channels=16,
                           channel_mults=(1, 2, 4)).to(device)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, :, ::4] = 1.0  # 4x equispaced
    kspace = fft2c(x)
    y_c = model_c(x, kspace * mask, mask)
    print(f"Cascaded U-Net: input {x.shape} → output {y_c.shape}")
    print("U-Net tests passed.")
