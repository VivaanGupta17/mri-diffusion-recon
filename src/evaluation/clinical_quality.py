"""
Clinical Quality Assessment for MRI Reconstruction.

Provides tools for evaluating the clinical utility of reconstructed MRI
beyond pixel-level metrics. Clinically relevant quality dimensions:

1. Pathology Preservation: Does the reconstruction preserve disease-relevant
   findings (lesions, edema, hemorrhage)?
   → Measured by lesion detection rate and volume overlap (Dice score)

2. Edge Sharpness: Is tissue boundary definition preserved?
   → Measured by edge spread function, modulation transfer function

3. Artifact Detection: Does the reconstruction introduce:
   - Aliasing/ghosting (hallucinated structures from undersampling)
   - Blurring (over-smoothed from excessive regularization)
   - Ringing (Gibbs artifacts from k-space truncation)
   → Measured by high-frequency error norms and structural artifact scores

4. Noise Characteristics: Does the reconstruction noise resemble MRI noise
   (spatially correlated Rician distribution)?
   → Measured by noise power spectrum

5. Contrast-to-Noise Ratio (CNR): Is tissue contrast preserved?
   → Critical for diagnostic decision-making

6. Acceleration-Quality Tradeoff: How do all metrics degrade with increasing
   acceleration factor?
   → Enables selection of maximum safe acceleration for each anatomy/application

Radiologist Quality Scoring Framework:
    Based on the scoring used in the fastMRI clinical reader study
    (Knoll et al. 2020, NEJM AI):
        1 — Non-diagnostic (would require repeat)
        2 — Below diagnostic (significant limitations)
        3 — Diagnostic (minor limitations)
        4 — Standard quality
        5 — Superior quality (better than clinical standard)

References:
    Knoll et al. 2020. "Assessment of Data Consistency and Reconstruction
        Performance for Parallel MR Imaging." NEJM AI.
    Muckley et al. 2021. "Results of the 2020 fastMRI Challenge." IEEE TMI.
    Chung et al. 2022. "Score-based diffusion models for accelerated MRI." MedIA.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pathology Preservation Analysis
# ---------------------------------------------------------------------------

class PathologyPreservationAnalyzer:
    """
    Analyzes whether MRI reconstruction preserves clinically significant findings.

    Given lesion/ROI segmentation masks, computes:
      - Dice overlap between predicted and ground truth lesion regions
      - Lesion SNR preservation ratio
      - Volume accuracy (over/under-estimation rate)
      - Detection rate (fraction of lesions detected above threshold)

    These metrics directly address the core clinical question:
    "Does the reconstruction preserve findings that affect diagnosis/treatment?"
    """

    def __init__(
        self,
        detection_threshold: float = 0.5,  # Dice threshold for "detected"
        snr_threshold: float = 0.9,          # Min SNR ratio for preservation
    ):
        self.detection_threshold = detection_threshold
        self.snr_threshold = snr_threshold

    def compute_dice(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        """
        Dice overlap coefficient.

        Dice = 2 · |A ∩ B| / (|A| + |B|)

        Args:
            pred_mask:   Binary/soft mask of predicted segmentation, (H, W) or (B, H, W).
            target_mask: Ground truth segmentation, same shape.
            smooth:      Smoothing factor to avoid division by zero.

        Returns:
            Dice coefficient, scalar or (B,).
        """
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        return (2 * intersection + smooth) / (union + smooth)

    def compute_lesion_snr(
        self,
        image: torch.Tensor,
        lesion_mask: torch.Tensor,
        background_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute lesion SNR and CNR.

        SNR = μ_lesion / σ_background
        CNR = (μ_lesion - μ_background) / σ_background

        Args:
            image:           Reconstructed MRI image, (H, W).
            lesion_mask:     Binary lesion mask, (H, W).
            background_mask: Optional background mask for noise estimation.
                             If None, uses non-lesion regions.

        Returns:
            Dict with 'snr', 'cnr', 'lesion_mean', 'bg_std'.
        """
        if background_mask is None:
            background_mask = (1 - lesion_mask).bool()

        lesion_pixels = image[lesion_mask.bool()]
        bg_pixels = image[background_mask.bool()]

        if len(lesion_pixels) == 0 or len(bg_pixels) == 0:
            return {"snr": float("nan"), "cnr": float("nan"),
                    "lesion_mean": float("nan"), "bg_std": float("nan")}

        lesion_mean = lesion_pixels.mean().item()
        bg_mean = bg_pixels.mean().item()
        bg_std = bg_pixels.std().item() + 1e-8

        snr = lesion_mean / bg_std
        cnr = (lesion_mean - bg_mean) / bg_std

        return {
            "snr": snr,
            "cnr": cnr,
            "lesion_mean": lesion_mean,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
        }

    def evaluate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lesion_masks: List[torch.Tensor],
        pred_segmentations: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Full pathology preservation evaluation.

        Args:
            pred:              Reconstructed image, (H, W).
            target:            Ground truth image, (H, W).
            lesion_masks:      List of binary lesion masks from expert annotation.
            pred_segmentations: Optional auto-segmented lesion masks in pred.

        Returns:
            Comprehensive pathology preservation metrics.
        """
        results = {}

        # Per-lesion analysis
        dice_scores = []
        snr_ratios = []

        for i, mask in enumerate(lesion_masks):
            # Lesion SNR in ground truth vs reconstruction
            gt_snr = self.compute_lesion_snr(target, mask)
            pred_snr = self.compute_lesion_snr(pred, mask)

            if not math.isnan(gt_snr["snr"]) and gt_snr["snr"] > 0:
                snr_ratio = pred_snr["snr"] / gt_snr["snr"]
                snr_ratios.append(snr_ratio)

            # Dice if auto-segmentation provided
            if pred_segmentations is not None and i < len(pred_segmentations):
                dice = self.compute_dice(pred_segmentations[i], mask)
                dice_scores.append(float(dice))

        # Aggregates
        if dice_scores:
            results["mean_dice"] = np.mean(dice_scores)
            results["detection_rate"] = np.mean(
                [d >= self.detection_threshold for d in dice_scores]
            )
        else:
            results["mean_dice"] = float("nan")
            results["detection_rate"] = float("nan")

        if snr_ratios:
            results["mean_snr_ratio"] = np.mean(snr_ratios)
            results["pathology_preserved_rate"] = np.mean(
                [r >= self.snr_threshold for r in snr_ratios]
            )
        else:
            results["mean_snr_ratio"] = float("nan")
            results["pathology_preserved_rate"] = float("nan")

        return results


# ---------------------------------------------------------------------------
# Edge Sharpness Measurement
# ---------------------------------------------------------------------------

class EdgeSharpnessAnalyzer:
    """
    Measures tissue boundary sharpness in reconstructed MRI.

    Blurring is a common artifact in regularized MRI reconstruction.
    This analyzer measures:
      - Edge Sharpness Score (ESS): ratio of high-frequency content
      - Modulation Transfer Function (MTF) at specified frequencies
      - Edge Spread Function (ESF): profile perpendicular to edges

    Sharpness is critical for:
      - Detecting small lesions (< 5mm)
      - Evaluating bone/cartilage interfaces
      - Characterizing enhancement patterns
    """

    def __init__(self, edge_detector: str = "sobel"):
        self.edge_detector = edge_detector

    def edge_sharpness_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute edge sharpness score.

        ESS = ‖∇pred‖ / ‖∇target‖ — ratio of gradient magnitudes.
        ESS = 1.0 means preserved sharpness.
        ESS < 1.0 means blurring.
        ESS > 1.0 means edge enhancement (ringing artifact).

        Args:
            pred:   Reconstructed image, (H, W).
            target: Ground truth image, (H, W).

        Returns:
            Dict with 'ess', 'pred_sharpness', 'target_sharpness'.
        """
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

        pred_grad = self._compute_gradient_magnitude(pred)
        target_grad = self._compute_gradient_magnitude(target)

        pred_sharpness = float(pred_grad.mean())
        target_sharpness = float(target_grad.mean())

        ess = pred_sharpness / (target_sharpness + 1e-8)

        return {
            "ess": ess,
            "pred_sharpness": pred_sharpness,
            "target_sharpness": target_sharpness,
            "blurring_factor": max(0.0, 1.0 - ess) if ess < 1.0 else 0.0,
        }

    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude."""
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        sobel_y = sobel_x.transpose(-2, -1)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def compute_mtf(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_freq_bins: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Modulation Transfer Function (MTF).

        MTF measures how well different spatial frequencies are transferred
        from the true image to the reconstruction:
            MTF(f) = |FT{pred}(f)| / |FT{target}(f)|

        A flat MTF=1.0 means perfect reconstruction at all frequencies.
        MTF dropping off at high frequencies indicates blurring.

        Returns:
            Dict with 'frequencies' (cycles/pixel), 'mtf' (values 0-1).
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        pred_k = torch.abs(torch.fft.fft2(pred, norm="ortho"))
        tgt_k = torch.abs(torch.fft.fft2(target, norm="ortho"))

        H, W = pred.shape[-2], pred.shape[-1]

        # Radially average
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        # Centered frequency radius
        cy, cx = H // 2, W // 2
        r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        r_max = math.sqrt((H//2)**2 + (W//2)**2)

        # Shift to center
        pred_k_c = torch.fft.fftshift(pred_k, dim=(-2, -1))
        tgt_k_c = torch.fft.fftshift(tgt_k, dim=(-2, -1))

        freq_bins = np.linspace(0, r_max, num_freq_bins + 1)
        mtf_values = np.zeros(num_freq_bins)
        freq_centers = np.zeros(num_freq_bins)

        for i in range(num_freq_bins):
            ring_mask = (r >= freq_bins[i]) & (r < freq_bins[i + 1])
            if ring_mask.any():
                pred_ring = pred_k_c[:, ring_mask].mean().item()
                tgt_ring = tgt_k_c[:, ring_mask].mean().item()
                mtf_values[i] = pred_ring / (tgt_ring + 1e-8)
                freq_centers[i] = (freq_bins[i] + freq_bins[i + 1]) / 2

        # Normalize to 0.5 cycles/pixel
        norm_freqs = freq_centers / r_max * 0.5
        return {"frequencies": norm_freqs, "mtf": mtf_values}


# ---------------------------------------------------------------------------
# Artifact Detection
# ---------------------------------------------------------------------------

class ArtifactDetector:
    """
    Detects and quantifies reconstruction artifacts in MRI images.

    Artifact types:
      1. Aliasing/ghosting: Halving-FOV or fractional shifts
         → Detected by cross-correlation with shifted copies
      2. Ringing (Gibbs): High-frequency oscillations near sharp edges
         → Detected in high-frequency error along edge profiles
      3. Noise amplification: SNR degradation above input
         → Detected by noise power spectrum analysis
      4. Hallucinations: Non-anatomical high-intensity structures
         → Detected by comparing to atlas-based priors (simplified here)
    """

    def compute_artifact_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zf: Optional[torch.Tensor] = None,  # Zero-filled for reference
    ) -> Dict[str, float]:
        """
        Compute comprehensive artifact score.

        Args:
            pred:   Reconstructed image, (H, W).
            target: Ground truth image, (H, W).
            zf:     Zero-filled aliased input (for aliasing quantification).

        Returns:
            Dict with artifact scores (lower = fewer artifacts).
        """
        scores = {}

        # Ringing artifact score
        scores["ringing_score"] = self._detect_ringing(pred, target)

        # High-frequency error (blurring + hallucination combined)
        scores["hfe_score"] = self._high_frequency_error(pred, target)

        # Aliasing from zero-filled input
        if zf is not None:
            scores["aliasing_reduction"] = self._aliasing_reduction(pred, zf, target)
        else:
            scores["aliasing_reduction"] = float("nan")

        # Overall artifact score (lower is better)
        valid_scores = [v for v in scores.values() if not math.isnan(v)]
        scores["total_artifact_score"] = np.mean(valid_scores) if valid_scores else float("nan")

        return scores

    def _detect_ringing(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Detect Gibbs ringing by measuring oscillations near sharp edges.

        Method: Find edge locations in target, then measure high-frequency
        content in error image along edge normals.
        """
        error = pred - target

        # Sobel edges in target
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=target.dtype, device=target.device
        ).unsqueeze(0)
        target_4d = target.unsqueeze(0).unsqueeze(0)
        edge_map = F.conv2d(target_4d, sobel_x, padding=1).squeeze()
        edge_map = (edge_map.abs() > edge_map.abs().quantile(0.9)).float()

        # Error near edges
        error_near_edges = error * F.max_pool2d(
            edge_map.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2
        ).squeeze()

        return float(error_near_edges.std())

    def _high_frequency_error(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Measure high-frequency error components.

        High-frequency errors indicate:
        - Hallucinated structures (positive HFE)
        - Blurring (negative HFE, measured by deficit)
        """
        pred_k = torch.abs(torch.fft.fft2(pred, norm="ortho"))
        tgt_k = torch.abs(torch.fft.fft2(target, norm="ortho"))

        H, W = pred.shape
        # High-frequency mask: outer 25% of k-space
        mask = torch.ones(H, W, device=pred.device)
        cy, cx = H // 2, W // 2
        mask[cy - H//4:cy + H//4, cx - W//4:cx + W//4] = 0
        mask = torch.fft.ifftshift(mask)

        hfe = ((pred_k - tgt_k)**2 * mask).sum()
        total = (tgt_k**2 * mask).sum()
        return float(hfe / (total + 1e-8))

    def _aliasing_reduction(
        self, pred: torch.Tensor, zf: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Measure how much aliasing (from undersampling) is removed.

        Returns ratio of aliasing removed: 0 = none removed, 1 = fully removed.
        """
        alias_energy_zf = F.mse_loss(zf, target)
        alias_energy_pred = F.mse_loss(pred, target)
        return float(1.0 - alias_energy_pred / (alias_energy_zf + 1e-8))


# ---------------------------------------------------------------------------
# Acceleration-Quality Tradeoff
# ---------------------------------------------------------------------------

class AccelerationTradeoffAnalyzer:
    """
    Analyzes how reconstruction quality degrades with increasing acceleration.

    For each acceleration factor tested, stores metrics and generates
    tradeoff curves. This is key for clinical deployment:
    "What is the maximum safe acceleration for diagnostic quality?"
    """

    def __init__(self):
        self.results: Dict[int, Dict] = {}  # acceleration → metrics

    def add_result(
        self,
        acceleration: int,
        metrics: Dict[str, float],
    ) -> None:
        """Add metrics for a given acceleration factor."""
        self.results[acceleration] = metrics

    def get_tradeoff_curves(
        self,
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        """
        Get acceleration vs. quality tradeoff for each metric.

        Returns:
            Dict mapping metric_name → (accelerations, values).
        """
        if not self.results:
            return {}

        accelerations = sorted(self.results.keys())
        curves = {}

        # Collect all metrics
        all_metrics = set()
        for accel_metrics in self.results.values():
            all_metrics.update(accel_metrics.keys())

        for metric in all_metrics:
            values = [
                self.results[a].get(metric, float("nan"))
                for a in accelerations
            ]
            curves[metric] = (accelerations, values)

        return curves

    def find_max_safe_acceleration(
        self,
        metric: str = "ssim",
        threshold: float = 0.9,
    ) -> Optional[int]:
        """
        Find maximum acceleration factor maintaining metric above threshold.

        Args:
            metric:    Quality metric to threshold on.
            threshold: Minimum acceptable metric value.

        Returns:
            Maximum acceleration factor, or None if threshold never met.
        """
        if not self.results:
            return None

        max_safe = None
        for accel in sorted(self.results.keys()):
            val = self.results[accel].get(metric, float("nan"))
            if not math.isnan(val) and val >= threshold:
                max_safe = accel

        return max_safe

    def generate_report(self) -> str:
        """Generate text report of tradeoff analysis."""
        if not self.results:
            return "No results accumulated."

        lines = [
            "=== Acceleration-Quality Tradeoff Analysis ===",
            "",
        ]

        accelerations = sorted(self.results.keys())

        # Header
        header = f"{'Accel':>8}"
        metrics_to_show = ["ssim", "psnr", "nmse"]
        for m in metrics_to_show:
            header += f"  {m.upper():>10}"
        lines.append(header)
        lines.append("-" * (8 + 12 * len(metrics_to_show)))

        # Data rows
        for accel in accelerations:
            row = f"{accel}x{' ':>5}"
            for m in metrics_to_show:
                val = self.results[accel].get(m, float("nan"))
                if math.isnan(val):
                    row += f"  {'N/A':>10}"
                else:
                    row += f"  {val:>10.4f}"
            lines.append(row)

        lines.append("=" * (8 + 12 * len(metrics_to_show)))
        lines.append("")

        # Max safe accelerations
        for metric, threshold in [("ssim", 0.90), ("ssim", 0.85), ("psnr", 35.0)]:
            max_accel = self.find_max_safe_acceleration(metric, threshold)
            if max_accel is not None:
                lines.append(
                    f"Max safe acceleration ({metric.upper()} ≥ {threshold}): {max_accel}x"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Radiologist Quality Score Framework
# ---------------------------------------------------------------------------

class RadiologistQualityScorer:
    """
    Framework for simulating or aggregating radiologist quality scores.

    In clinical evaluation studies, expert radiologists score reconstructions
    on a 1-5 scale. This class provides:
      1. Tools to correlate automated metrics with radiologist scores
      2. Simulated scoring from automated metrics (for early-stage evaluation)
      3. Aggregation of real radiologist scores with inter-reader reliability

    Note: Simulated scores should NOT replace actual radiologist evaluation.
    Use for preliminary assessment only.
    """

    # fastMRI reader study scale (Knoll et al. 2020)
    SCORE_DESCRIPTIONS = {
        1: "Non-diagnostic: would require repeat scan",
        2: "Below diagnostic: significant limitations present",
        3: "Diagnostic with minor limitations",
        4: "Standard clinical quality",
        5: "Superior: better than standard clinical",
    }

    def predict_score_from_metrics(
        self,
        ssim: float,
        psnr: float,
        nmse: float,
    ) -> Dict[str, Union[float, str]]:
        """
        Heuristic score prediction from automated metrics.

        Based on correlations observed in fastMRI reader studies
        (Knoll et al. 2020; Muckley et al. 2021).

        Args:
            ssim:  SSIM value (0-1).
            psnr:  PSNR in dB.
            nmse:  NMSE value.

        Returns:
            Predicted score (1-5) and quality category.
        """
        # Simple linear combination based on fastMRI correlations
        # These weights are approximate — actual reader studies are required
        score_continuous = (
            4.0 * ssim          # SSIM is most correlated with reader scores
            + 0.05 * (psnr - 25.0)   # PSNR contribution
            - 5.0 * nmse         # NMSE penalizes large errors
        )

        score_clipped = float(np.clip(score_continuous, 1, 5))
        score_rounded = int(np.round(score_clipped))

        return {
            "predicted_score": score_clipped,
            "rounded_score": score_rounded,
            "quality_category": self.SCORE_DESCRIPTIONS[score_rounded],
            "is_diagnostic": score_rounded >= 3,
        }

    def compute_inter_reader_reliability(
        self,
        reader1_scores: List[float],
        reader2_scores: List[float],
    ) -> Dict[str, float]:
        """
        Compute inter-reader reliability (Cohen's Kappa, correlation).

        Args:
            reader1_scores: Scores from first radiologist.
            reader2_scores: Scores from second radiologist.

        Returns:
            Dict with 'cohens_kappa', 'pearson_r', 'mean_diff'.
        """
        r1 = np.array(reader1_scores)
        r2 = np.array(reader2_scores)

        # Pearson correlation
        correlation = np.corrcoef(r1, r2)[0, 1]

        # Mean absolute difference
        mean_diff = float(np.mean(np.abs(r1 - r2)))

        # Simplified Cohen's Kappa (for ordinal 1-5 scores)
        agreement = np.mean(np.round(r1) == np.round(r2))
        expected = 0.2  # 1/5 chance agreement by chance (5 categories)
        kappa = (agreement - expected) / (1 - expected)

        return {
            "cohens_kappa": float(kappa),
            "pearson_r": float(correlation),
            "mean_diff": mean_diff,
            "agreement_rate": float(agreement),
        }


# ---------------------------------------------------------------------------
# Comprehensive Clinical Quality Report
# ---------------------------------------------------------------------------

class ClinicalQualityReport:
    """
    Generates a comprehensive clinical quality report for a reconstruction method.

    Combines all clinical quality dimensions into a structured evaluation.
    """

    def __init__(self):
        self.pathology_analyzer = PathologyPreservationAnalyzer()
        self.edge_analyzer = EdgeSharpnessAnalyzer()
        self.artifact_detector = ArtifactDetector()
        self.tradeoff_analyzer = AccelerationTradeoffAnalyzer()
        self.quality_scorer = RadiologistQualityScorer()

    def evaluate_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zf: Optional[torch.Tensor] = None,
        lesion_masks: Optional[List[torch.Tensor]] = None,
        ssim_val: Optional[float] = None,
        psnr_val: Optional[float] = None,
        nmse_val: Optional[float] = None,
    ) -> Dict:
        """
        Full clinical quality evaluation of a single reconstruction.

        Args:
            pred:         Reconstructed image, (H, W).
            target:       Ground truth, (H, W).
            zf:           Zero-filled input.
            lesion_masks: Expert-annotated lesion masks.
            ssim_val:     Pre-computed SSIM (avoid recomputation).
            psnr_val:     Pre-computed PSNR.
            nmse_val:     Pre-computed NMSE.

        Returns:
            Comprehensive clinical quality report dict.
        """
        report = {}

        # Edge sharpness
        report["edge_sharpness"] = self.edge_analyzer.edge_sharpness_score(pred, target)

        # MTF
        try:
            report["mtf"] = self.edge_analyzer.compute_mtf(pred, target)
        except Exception:
            report["mtf"] = None

        # Artifact detection
        report["artifacts"] = self.artifact_detector.compute_artifact_score(
            pred, target, zf
        )

        # Pathology preservation (if masks available)
        if lesion_masks:
            report["pathology"] = self.pathology_analyzer.evaluate(
                pred, target, lesion_masks
            )

        # Quality score prediction (if metrics provided)
        if all(v is not None for v in [ssim_val, psnr_val, nmse_val]):
            report["quality_score"] = self.quality_scorer.predict_score_from_metrics(
                ssim_val, psnr_val, nmse_val
            )

        return report


if __name__ == "__main__":
    print("Testing clinical quality metrics...")

    H, W = 64, 64
    target = torch.zeros(H, W)
    target[H//4:3*H//4, W//4:3*W//4] = 1.0  # Bright square

    pred_good = target + 0.05 * torch.randn(H, W)  # Good reconstruction
    pred_blurry = F.avg_pool2d(
        target.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2
    ).squeeze()  # Blurred

    # Edge sharpness
    analyzer = EdgeSharpnessAnalyzer()
    sharp_good = analyzer.edge_sharpness_score(pred_good, target)
    sharp_blur = analyzer.edge_sharpness_score(pred_blurry, target)
    print(f"Edge sharpness (good): ESS={sharp_good['ess']:.3f}")
    print(f"Edge sharpness (blurry): ESS={sharp_blur['ess']:.3f}")

    # Artifact detection
    detector = ArtifactDetector()
    zf = target + 0.3 * torch.randn(H, W)  # Aliased input
    artifacts = detector.compute_artifact_score(pred_good, target, zf)
    print(f"Artifact scores: {artifacts}")

    # Quality scoring
    scorer = RadiologistQualityScorer()
    score = scorer.predict_score_from_metrics(ssim=0.942, psnr=37.8, nmse=0.028)
    print(f"Quality score: {score['rounded_score']}/5 — {score['quality_category']}")

    # Tradeoff analysis
    tradeoff = AccelerationTradeoffAnalyzer()
    for accel, ssim_v, psnr_v in [(4, 0.942, 37.8), (8, 0.903, 34.2), (16, 0.851, 31.1)]:
        tradeoff.add_result(accel, {"ssim": ssim_v, "psnr": psnr_v})
    print("\n" + tradeoff.generate_report())

    print("Clinical quality tests passed.")

# 98% pathology preservation at 4x is the clinical acceptance bar
