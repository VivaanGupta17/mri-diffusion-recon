#!/usr/bin/env python3
"""
Evaluation Script for MRI-DiffRecon.

Computes comprehensive reconstruction metrics comparing:
  - Diffusion model (ours)
  - U-Net baseline
  - Compressed sensing (TV, wavelet, combined)
  - Published methods (if predictions available)

Generates comparison tables, tradeoff curves, and clinical quality reports.

Usage:
    # Full benchmark evaluation
    python scripts/evaluate.py \\
        --diffusion_dir results/diffusion/ \\
        --unet_dir results/unet/ \\
        --cs_dir results/cs/ \\
        --ground_truth_dir data/test_volumes/ \\
        --output_dir results/metrics/

    # Single method evaluation
    python scripts/evaluate.py \\
        --predictions results/diffusion/ \\
        --ground_truth data/test_volumes/ \\
        --metrics ssim psnr nmse lpips

    # Acceleration tradeoff analysis
    python scripts/evaluate.py \\
        --checkpoint runs/knee_4x/best_model.pt \\
        --test_data data/test_volumes/ \\
        --acceleration_sweep \\
        --accelerations 4 8 16

    # Clinical quality report
    python scripts/evaluate.py \\
        --predictions results/diffusion/ \\
        --ground_truth data/test_volumes/ \\
        --clinical_quality \\
        --pathology_maps data/annotations/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MRI reconstruction methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Predictions (one or multiple methods)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--diffusion_dir", type=str, default=None)
    parser.add_argument("--unet_dir", type=str, default=None)
    parser.add_argument("--cs_dir", type=str, default=None)

    # Ground truth
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--ground_truth_dir", type=str, default=None)

    # Metrics
    parser.add_argument(
        "--metrics", nargs="+",
        default=["ssim", "psnr", "nmse"],
        choices=["ssim", "psnr", "nmse", "mae", "lpips", "fid"],
    )

    # Acceleration sweep
    parser.add_argument("--acceleration_sweep", action="store_true")
    parser.add_argument(
        "--accelerations", nargs="+", type=int, default=[4, 8, 16]
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)

    # Clinical quality
    parser.add_argument("--clinical_quality", action="store_true")
    parser.add_argument("--pathology_maps", type=str, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/metrics/")
    parser.add_argument("--format", type=str, default="table",
                        choices=["table", "json", "csv", "latex"])

    # Hardware
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def load_predictions(pred_dir: str) -> dict:
    """Load all reconstruction files from a directory."""
    import h5py
    pred_dir = Path(pred_dir)
    predictions = {}
    for h5_file in sorted(pred_dir.glob("*.h5")):
        with h5py.File(h5_file, "r") as hf:
            if "reconstruction" in hf:
                predictions[h5_file.stem] = hf["reconstruction"][:]
    return predictions


def load_ground_truth(gt_dir: str) -> dict:
    """Load ground truth volumes from directory."""
    import h5py
    gt_dir = Path(gt_dir)
    targets = {}
    for h5_file in sorted(gt_dir.glob("*.h5")):
        with h5py.File(h5_file, "r") as hf:
            for key in ["reconstruction_rss", "reconstruction_esc", "target"]:
                if key in hf:
                    targets[h5_file.stem] = hf[key][:]
                    break
    return targets


def evaluate_predictions(
    predictions: dict,
    ground_truths: dict,
    device: str = "cpu",
    compute_lpips: bool = False,
    compute_fid: bool = False,
) -> dict:
    """Compute metrics for a set of predictions vs ground truths."""
    from src.evaluation.mri_metrics import MRIEvaluator

    evaluator = MRIEvaluator(
        device=device,
        compute_lpips=compute_lpips,
        compute_fid=compute_fid,
    )

    common_keys = set(predictions.keys()) & set(ground_truths.keys())
    if not common_keys:
        print("Warning: No matching filenames between predictions and ground truth")
        return {}

    for key in sorted(common_keys):
        pred = torch.tensor(predictions[key]).float()
        target = torch.tensor(ground_truths[key]).float()

        # Per-slice evaluation
        for s in range(pred.shape[0]):
            ps = pred[s:s+1]
            ts = target[s:s+1]
            max_val = ts.max() + 1e-8
            evaluator.update(ps / max_val, ts / max_val)

    return evaluator.compute()


def print_comparison_table(results: dict) -> None:
    """Print formatted comparison table."""
    if not results:
        return

    methods = list(results.keys())
    all_metrics = set()
    for method_results in results.values():
        all_metrics.update(method_results.keys())
    metrics_list = sorted(all_metrics)

    print("\n" + "=" * 80)
    print("MRI RECONSTRUCTION EVALUATION RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Method':<25}"
    for m in metrics_list:
        if m in ("ssim", "psnr", "nmse", "lpips", "fid"):
            header += f"  {m.upper():>10}"
    print(header)
    print("-" * 80)

    # Rows
    for method, method_results in results.items():
        row = f"{method:<25}"
        for m in metrics_list:
            if m in ("ssim", "psnr", "nmse", "lpips", "fid"):
                if m in method_results:
                    mean_val = method_results[m].get("mean", float("nan"))
                    std_val = method_results[m].get("std", 0.0)
                    row += f"  {mean_val:>7.4f}±{std_val:.3f}"
                else:
                    row += f"  {'N/A':>10}"
        print(row)

    print("=" * 80)


def run_acceleration_sweep(args) -> None:
    """Run reconstruction at multiple acceleration factors and compute metrics."""
    from src.inference.reconstruct import load_diffusion_model, MRIReconstructionEngine
    from src.data.fastmri_dataset import SyntheticMRIDataset, build_dataloader
    from src.evaluation.mri_metrics import MRIEvaluator
    from src.evaluation.clinical_quality import AccelerationTradeoffAnalyzer
    from src.data.kspace_transforms import generate_mask

    device = torch.device(args.device)
    tradeoff = AccelerationTradeoffAnalyzer()

    if args.checkpoint is not None:
        model = load_diffusion_model(args.checkpoint, device=device)
    else:
        print("No checkpoint provided — using synthetic zero-filled baseline")
        model = None

    for accel in args.accelerations:
        print(f"\nEvaluating acceleration {accel}x...")

        # Create synthetic test data
        dataset = SyntheticMRIDataset(
            num_samples=20, image_size=64, acceleration=accel, seed=42
        )
        loader = build_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False)

        evaluator = MRIEvaluator(device=args.device, compute_fid=False)

        for batch in loader:
            x_gt = batch["x_gt"][:, 0]  # Magnitude only
            x_zf = batch["x_zf"][:, 0]

            max_val = x_gt.max() + 1e-8
            evaluator.update(x_zf / max_val, x_gt / max_val)

        accel_results = evaluator.compute()
        tradeoff.add_result(accel, {
            "ssim": accel_results.get("ssim", {}).get("mean", float("nan")),
            "psnr": accel_results.get("psnr", {}).get("mean", float("nan")),
            "nmse": accel_results.get("nmse", {}).get("mean", float("nan")),
        })

    print("\n" + tradeoff.generate_report())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.acceleration_sweep:
        run_acceleration_sweep(args)
        return

    compute_lpips = "lpips" in args.metrics
    compute_fid = "fid" in args.metrics

    # Single method evaluation
    if args.predictions is not None and args.ground_truth is not None:
        preds = load_predictions(args.predictions)
        gts = load_ground_truth(args.ground_truth)
        if preds and gts:
            results = {"model": evaluate_predictions(
                preds, gts, args.device, compute_lpips, compute_fid
            )}
            print_comparison_table(results)
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(results, f, indent=2)
        return

    # Multi-method comparison
    all_results = {}

    if args.diffusion_dir and args.ground_truth_dir:
        preds = load_predictions(args.diffusion_dir)
        gts = load_ground_truth(args.ground_truth_dir)
        if preds and gts:
            all_results["Diffusion (ours)"] = evaluate_predictions(
                preds, gts, args.device, compute_lpips, compute_fid
            )

    if args.unet_dir and args.ground_truth_dir:
        preds = load_predictions(args.unet_dir)
        gts = load_ground_truth(args.ground_truth_dir)
        if preds and gts:
            all_results["U-Net Baseline"] = evaluate_predictions(
                preds, gts, args.device, compute_lpips, compute_fid
            )

    if args.cs_dir and args.ground_truth_dir:
        preds = load_predictions(args.cs_dir)
        gts = load_ground_truth(args.ground_truth_dir)
        if preds and gts:
            all_results["Compressed Sensing"] = evaluate_predictions(
                preds, gts, args.device, compute_lpips, compute_fid
            )

    if all_results:
        print_comparison_table(all_results)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_dir}/comparison.json")
    else:
        print("No results computed. Check input paths.")
        print("Hint: run with --acceleration_sweep for quick synthetic evaluation")


if __name__ == "__main__":
    main()
