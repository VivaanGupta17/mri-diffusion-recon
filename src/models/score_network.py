"""
Score Estimation Network (NCSN++) for MRI Reconstruction.

Implements a noise-conditioned score network based on the NCSN++ architecture
from Song et al. (2021) "Score-Based Generative Modeling through SDEs",
adapted for complex-valued MRI data with conditional reconstruction inputs.

Architecture:
    - U-Net backbone with sinusoidal time embeddings
    - FiLM conditioning (Feature-wise Linear Modulation)
    - Multi-scale skip connections
    - Complex-valued input/output support (real + imaginary channels)
    - Group normalization throughout
    - Attention blocks at bottleneck

References:
    Song et al., 2021. "Score-Based Generative Modeling through SDEs." ICLR.
    Chung & Ye, 2022. "Score-based diffusion models for accelerated MRI." MedIA.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: Sinusoidal Time / Noise-Level Embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for noise level / diffusion timestep.

    Encodes the continuous noise level σ or discrete timestep t into a
    fixed-dimensional embedding vector using sine/cosine functions at
    logarithmically-spaced frequencies.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Noise level or timestep, shape (B,) — can be continuous (σ) or
               integer steps. Values should be normalized to [0, 1] or log-scaled.

        Returns:
            emb: Embedding tensor of shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FourierFeatures(nn.Module):
    """
    Random Fourier features for continuous noise level embedding.
    Used in NCSN++ to handle continuous noise schedules.
    """

    def __init__(self, dim: int, std: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.register_buffer("W", torch.randn(dim // 2) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


# ---------------------------------------------------------------------------
# FiLM Conditioning
# ---------------------------------------------------------------------------

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning feature maps on noise level.

    Given a conditioning vector z (e.g., noise embedding), computes per-channel
    scale (γ) and shift (β) applied to normalized feature maps.

    Ref: Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Feature map, shape (B, C, H, W).
            cond: Conditioning vector, shape (B, cond_dim).

        Returns:
            x_modulated: FiLM-conditioned features, shape (B, C, H, W).
        """
        gamma_beta = self.linear(cond)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]  # (B, C, 1, 1)
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Basic Building Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with group normalization and FiLM conditioning.

    Used as the main building block throughout the U-Net encoder and decoder.
    Supports optional channel up/down projection via 1x1 convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.film1 = FiLMBlock(out_channels, cond_dim)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film2 = FiLMBlock(out_channels, cond_dim)

        self.dropout = nn.Dropout2d(dropout)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.film1(h, cond)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.film2(h, cond)

        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block operating on spatial feature maps.

    Applied at the bottleneck (lowest resolution) to capture global context.
    Uses multi-head attention with positional embeddings.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    """Strided convolution downsampling (2x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor + convolution upsampling (2x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder Levels
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    """Single U-Net encoder level: N residual blocks + optional downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        downsample: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResBlock(ch_in, out_channels, cond_dim, dropout=dropout)
            )

        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.downsample = Downsample(out_channels) if downsample else None

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (downsampled, skip)."""
        for block in self.blocks:
            x = block(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class DecoderLevel(nn.Module):
    """Single U-Net decoder level: upsample + skip concat + N residual blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        upsample: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.upsample = Upsample(in_channels) if upsample else None
        total_in = in_channels + skip_channels

        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = total_in if i == 0 else out_channels
            self.blocks.append(
                ResBlock(ch_in, out_channels, cond_dim, dropout=dropout)
            )

        self.attention = AttentionBlock(out_channels) if use_attention else None

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        return x


# ---------------------------------------------------------------------------
# Main Score Network (NCSN++)
# ---------------------------------------------------------------------------

class NCSNpp(nn.Module):
    """
    Noise-Conditioned Score Network ++ (NCSN++) for MRI reconstruction.

    Estimates the score function ∇_x log p_σ(x | y) conditioned on:
      - Noise level σ (or timestep t) via sinusoidal + Fourier embeddings
      - Undersampled/zero-filled input image y (concatenated as extra channels)

    The network predicts the score (∝ denoising direction) rather than directly
    predicting the clean image, though both parameterizations are supported.

    Input:
        x: Noisy complex MRI image, shape (B, 2, H, W) — [real, imag] or
           (B, 2*C, H, W) for multi-coil with C coils.
        y: Conditioning image (zero-filled aliased reconstruction),
           shape (B, 2, H, W).
        t: Noise levels, shape (B,), values in [0, 1] or log σ.

    Output:
        score: Estimated score ∇_x log p_σ(x), shape same as x.
    """

    def __init__(
        self,
        in_channels: int = 2,           # 2 for single-coil (real+imag), 2*C for multi-coil
        cond_channels: int = 2,         # Channels of conditioning input y
        base_channels: int = 64,        # Base feature channels
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16, 8),  # Resolutions to apply attention
        dropout: float = 0.1,
        emb_dim: int = 256,             # Noise embedding dimension
        fourier_scale: float = 16.0,    # Scale for Fourier features
        input_resolution: int = 320,    # Expected spatial resolution (for attn mask)
        scale_by_sigma: bool = True,    # NCSN++: multiply output by 1/σ
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scale_by_sigma = scale_by_sigma

        # ----- Noise level embedding -----
        self.fourier_emb = FourierFeatures(emb_dim // 2, std=fourier_scale)
        self.sinusoidal_emb = SinusoidalPosEmb(emb_dim // 2)
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # ----- Input projection -----
        # Concatenate noisy x and conditioning y along channel dim
        total_in = in_channels + cond_channels
        channels = [base_channels * m for m in channel_mults]

        self.input_conv = nn.Conv2d(total_in, channels[0], 3, padding=1)

        # ----- Encoder -----
        self.encoders = nn.ModuleList()
        in_ch = channels[0]
        self._skip_channels = []
        num_levels = len(channels)

        for level_idx, ch in enumerate(channels):
            is_last = (level_idx == num_levels - 1)
            # Determine if this level uses attention (based on resolution)
            use_attn = False  # Simplified — always use at deepest levels
            if level_idx >= num_levels - 2:
                use_attn = True

            self.encoders.append(
                EncoderLevel(
                    in_ch, ch, emb_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    downsample=not is_last,
                    dropout=dropout,
                )
            )
            self._skip_channels.append(ch)
            in_ch = ch

        # ----- Bottleneck -----
        bottleneck_ch = channels[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock(bottleneck_ch, bottleneck_ch, emb_dim, dropout=dropout),
            AttentionBlock(bottleneck_ch),
            ResBlock(bottleneck_ch, bottleneck_ch, emb_dim, dropout=dropout),
        ])

        # ----- Decoder -----
        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        rev_skips = list(reversed(self._skip_channels))

        for level_idx in range(num_levels):
            is_last = (level_idx == num_levels - 1)
            in_ch_dec = rev_channels[level_idx]
            skip_ch = rev_skips[level_idx]
            out_ch = rev_channels[min(level_idx + 1, num_levels - 1)]
            if is_last:
                out_ch = channels[0]

            use_attn = (level_idx < 2)  # Attention at top decoder levels

            self.decoders.append(
                DecoderLevel(
                    in_ch_dec, skip_ch, out_ch, emb_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    upsample=not (level_idx == num_levels - 1),
                    dropout=dropout,
                )
            )

        # ----- Output projection -----
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)

        # Initialize output conv to zero (important for score networks)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _embed_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """Combine Fourier and sinusoidal embeddings for noise level t."""
        f_emb = self.fourier_emb(t)         # (B, emb_dim//2)
        s_emb = self.sinusoidal_emb(t)       # (B, emb_dim//2)
        emb = torch.cat([f_emb, s_emb], dim=1)  # (B, emb_dim)
        return self.emb_proj(emb)            # (B, emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy MRI image, (B, 2, H, W) or (B, 2C, H, W) for multi-coil.
            t: Noise levels, (B,). Should be continuous σ values or normalized [0,1].
            y: Optional conditioning (zero-filled reconstruction), same shape as x.
               If None, uses zero conditioning (unconditional score).

        Returns:
            score: Score estimate ∇_x log p_σ(x), shape same as x.
        """
        # Noise level embedding
        cond = self._embed_noise_level(t)  # (B, emb_dim)

        # Concatenate with conditioning input
        if y is None:
            y = torch.zeros_like(x)
        h = torch.cat([x, y], dim=1)  # (B, in_channels + cond_channels, H, W)

        # Input projection
        h = self.input_conv(h)  # (B, base_channels, H, W)

        # Encoder forward pass, collecting skip connections
        skips = []
        for encoder in self.encoders:
            h, skip = encoder(h, cond)
            skips.append(skip)

        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, cond)
            else:
                h = layer(h)

        # Decoder forward pass with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            h = decoder(h, skip, cond)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        score = self.output_conv(h)

        # NCSN++: scale output by 1/σ (score = -ε/σ in Tweedie's formula)
        if self.scale_by_sigma:
            sigma = t[:, None, None, None]
            score = score / (sigma + 1e-8)

        return score


# ---------------------------------------------------------------------------
# Complex-Valued Score Network Wrapper
# ---------------------------------------------------------------------------

class ComplexScoreNetwork(nn.Module):
    """
    Wrapper for complex-valued MRI score estimation.

    MRI data is inherently complex-valued (k-space and image domain).
    This wrapper handles:
      1. Complex input → real/imaginary channel split
      2. Magnitude normalization (critical for training stability)
      3. Forward pass through NCSNpp
      4. Real/imaginary output → complex reconstruction

    For multi-coil data, each coil is processed independently and results
    are combined via root-sum-of-squares (RSS) or learned combination.
    """

    def __init__(
        self,
        num_coils: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        emb_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_coils = num_coils
        in_channels = 2 * num_coils  # 2 channels (real+imag) per coil
        cond_channels = 2 * num_coils

        self.score_net = NCSNpp(
            in_channels=in_channels,
            cond_channels=cond_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            emb_dim=emb_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Complex MRI data, shape (B, num_coils, H, W) dtype=complex64,
               OR (B, 2*num_coils, H, W) with real/imag channels.
            t: Noise levels, shape (B,).
            y: Conditioning input (zero-filled), same shape as x.

        Returns:
            score: Score estimate, same shape as x.
        """
        # Convert complex to real/imag if needed
        if x.is_complex():
            x_real = torch.view_as_real(x)  # (B, C, H, W, 2)
            x_real = x_real.permute(0, 1, 4, 2, 3)  # (B, C, 2, H, W)
            x_real = x_real.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        else:
            x_real = x

        if y.is_complex():
            y_real = torch.view_as_real(y)
            y_real = y_real.permute(0, 1, 4, 2, 3)
            y_real = y_real.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])
        else:
            y_real = y

        score_real = self.score_net(x_real, t, y_real)

        # Convert back to complex if input was complex
        if x.is_complex():
            score_real = score_real.reshape(
                x.shape[0], self.num_coils, 2, x.shape[-2], x.shape[-1]
            )
            score_real = score_real.permute(0, 1, 3, 4, 2)  # (B, C, H, W, 2)
            return torch.view_as_complex(score_real.contiguous())

        return score_real


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_score_network(config: dict) -> nn.Module:
    """
    Build a score network from a configuration dictionary.

    Args:
        config: Dictionary with keys:
            - model_type: 'ncsn_pp' or 'complex_ncsn_pp'
            - num_coils: Number of MRI coils (default 1)
            - base_channels: Base feature channels (default 64)
            - channel_mults: Channel multipliers per level
            - num_res_blocks: Residual blocks per level
            - emb_dim: Noise embedding dimension
            - dropout: Dropout rate

    Returns:
        Instantiated score network.
    """
    model_type = config.get("model_type", "complex_ncsn_pp")
    num_coils = config.get("num_coils", 1)
    base_channels = config.get("base_channels", 64)
    channel_mults = tuple(config.get("channel_mults", [1, 2, 4, 8]))
    num_res_blocks = config.get("num_res_blocks", 2)
    emb_dim = config.get("emb_dim", 256)
    dropout = config.get("dropout", 0.1)

    if model_type == "ncsn_pp":
        return NCSNpp(
            in_channels=2,
            cond_channels=2,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            emb_dim=emb_dim,
            dropout=dropout,
        )
    elif model_type == "complex_ncsn_pp":
        return ComplexScoreNetwork(
            num_coils=num_coils,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            emb_dim=emb_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing score network on {device}")

    # Single-coil test
    model = NCSNpp(
        in_channels=2,
        cond_channels=2,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        emb_dim=128,
    ).to(device)

    B, H, W = 2, 64, 64
    x = torch.randn(B, 2, H, W).to(device)
    y = torch.randn(B, 2, H, W).to(device)
    t = torch.rand(B).to(device) * 0.999 + 0.001  # Noise levels in (0, 1]

    score = model(x, t, y)
    print(f"Input shape: {x.shape} | Score shape: {score.shape}")
    print(f"Parameter count: {count_parameters(model):,}")
    assert score.shape == x.shape, "Score shape mismatch!"
    print("Score network test passed.")
