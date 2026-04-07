#!/usr/bin/env python3
"""
MRI Reconstruction Inference Script.

Reconstructs MRI volumes from undersampled k-space data using the trained
diffusion model or baseline methods.

Usage:
    # Single volume reconstruction
    python scripts/reconstruct.py \\
        --checkpoint runs/knee_4x/best_model.pt \\
        --input data/sample.h5 \\
        --output results/recon.h5 \\
        --method pc --num_steps 1000

    # Fast inference with DDIM (50 steps, ~10x speedup)
    python scripts/reconstruct.py \\
        --checkpoint runs/knee_4x/best_model.pt \\
        --input data/sample.h5 \\
        --output results/recon_fast.h5 \\
        --method ddim --num_steps 50

    # Batch reconstruction of directory
    python scripts/reconstruct.py \\
        --checkpoint runs/knee_4x/best_model.pt \\
        --input_dir data/test_volumes/ \\
        --output_dir results/ \\
        --method pc --num_steps 1000

    # Baseline methods
    python scripts/reconstruct.py \\
        --method cs --cs_method combined \\
        --input data/sample.h5 --output results/cs_recon.h5

    # Evaluate reconstruction quality (compare to ground truth)
    python scripts/reconstruct.py \\
        --checkpoint runs/knee_4x/best_model.pt \\
        --input data/sample.h5 --output results/recon.h5 \\
        --ground_truth data/sample.h5 --compute_metrics
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="MRI reconstruction inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input .h5 file (single volume)"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Input directory (batch mode)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .h5 file (single volume)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/",
        help="Output directory (batch mode)"
    )

    # Model
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--no_ema", action="store_true",
        help="Use raw model weights instead of EMA"
    )

    # Sampling
    parser.add_argument(
        "--method", type=str, default="pc",
        choices=["pc", "em", "ddim", "unet", "cs"],
        help="Reconstruction method"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--num_corrector_steps", type=int, default=1,
        help="Corrector steps per predictor step (PC only)"
    )

    # CS-specific
    parser.add_argument(
        "--cs_method", type=str, default="combined",
        choices=["tv", "wavelet", "combined"],
        help="Compressed sensing method"
    )
    parser.add_argument(
        "--lambda_tv", type=float, default=0.005,
        help="TV regularization weight (CS)"
    )
    parser.add_argument(
        "--lambda_wav", type=float, default=0.005,
        help="Wavelet regularization weight (CS)"
    )

    # Acquisition
    parser.add_argument(
        "--acceleration", type=int, default=4,
        help="Undersampling acceleration factor"
    )
    parser.add_argument(
        "--center_fractions", type=float, default=0.08,
        help="Fraction of center k-space lines (ACS)"
    )
    parser.add_argument(
        "--mask_type", type=str, default="random",
        choices=["random", "equispaced", "poisson"],
        help="Undersampling mask type"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Inference device"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable mixed precision inference"
    )

    # Evaluation
    parser.add_argument(
        "--ground_truth", type=str, default=None,
        help="Ground truth .h5 file for metric computation"
    )
    parser.add_argument(
        "--compute_metrics", action="store_true",
        help="Compute reconstruction metrics against ground truth"
    )

    # Output format
    parser.add_argument(
        "--save_png", action="store_true",
        help="Also save PNG visualizations of reconstructions"
    )

    return parser.parse_args()


def reconstruct_diffusion(args):
    """Run diffusion model reconstruction."""
    from src.inference.reconstruct import reconstruct_from_h5, batch_reconstruct

    use_ema = not args.no_ema
    use_amp = not args.no_amp

    if args.input is not None:
        # Single volume
        if args.output is None:
            args.output = str(Path(args.input).stem) + "_recon.h5"

        print(f"Reconstructing: {args.input}")
        t0 = time.time()

        results = reconstruct_from_h5(
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            method=args.method,
            num_steps=args.num_steps,
            acceleration=args.acceleration,
            center_fractions=args.center_fractions,
            mask_type=args.mask_type,
            device=args.device,
            verbose=True,
        )

        print(f"\nReconstruction complete in {time.time()-t0:.1f}s")
        print(f"Output: {args.output}")

    elif args.input_dir is not None:
        # Batch mode
        print(f"Batch reconstruction from: {args.input_dir}")
        results = batch_reconstruct(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            method=args.method,
            num_steps=args.num_steps,
            acceleration=args.acceleration,
            device=args.device,
        )
        print(f"\nProcessed {len(results)} volumes → {args.output_dir}")

    else:
        print("Error: Provide --input or --input_dir")
        sys.exit(1)


def reconstruct_cs(args):
    """Run compressed sensing reconstruction."""
    import h5py
    import numpy as np
    import torch
    from src.models.compressed_sensing import CSReconBenchmark
    from src.data.kspace_transforms import generate_mask

    print(f"CS reconstruction ({args.cs_method}): {args.input}")

    cs_method_map = {"tv": "tv_admm", "wavelet": "wavelet_fista", "combined": "combined_cs"}
    solver = CSReconBenchmark(
        method=cs_method_map[args.cs_method],
        lambda_tv=args.lambda_tv,
        lambda_wav=args.lambda_wav,
        device=args.device,
    )

    with h5py.File(args.input, "r") as hf:
        kspace_np = hf["kspace"][:]

    # Convert to tensor
    kspace = torch.tensor(kspace_np, dtype=torch.complex64)
    if kspace.dim() == 2:
        kspace = kspace.unsqueeze(0)

    num_slices = kspace.shape[0]
    H, W = kspace.shape[-2], kspace.shape[-1]

    # Generate mask
    mask, accel = generate_mask(
        (1, H, W), args.acceleration, args.center_fractions, args.mask_type
    )
    mask_2d = mask.unsqueeze(0)

    print(f"  Slices: {num_slices}, Resolution: {H}×{W}, Accel: {accel:.1f}x")

    recons = []
    t0 = time.time()

    for i in range(num_slices):
        # Stack real/imag
        k = kspace[i]
        k_ri = torch.stack([k.real, k.imag], dim=0).unsqueeze(0)  # (1, 2, H, W)
        k_masked = k_ri * mask_2d
        recon = solver.reconstruct(k_masked, mask_2d)
        # Magnitude
        mag = torch.sqrt(recon[0, 0]**2 + recon[0, 1]**2).cpu().numpy()
        recons.append(mag)

        if (i + 1) % 10 == 0:
            print(f"  Slice {i+1}/{num_slices} ({time.time()-t0:.1f}s elapsed)")

    recons = np.stack(recons)
    output_path = args.output or (Path(args.input).stem + f"_cs_{args.cs_method}.h5")

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("reconstruction", data=recons)
        hf.attrs["method"] = f"cs_{args.cs_method}"
        hf.attrs["acceleration"] = accel

    print(f"\nCS reconstruction complete in {time.time()-t0:.1f}s")
    print(f"Output: {output_path}")


def compute_metrics_comparison(args):
    """Compute reconstruction quality metrics."""
    import h5py
    import numpy as np
    import torch
    from src.evaluation.mri_metrics import MRIEvaluator

    print("\nComputing reconstruction metrics...")

    with h5py.File(args.output or args.output_dir, "r") as hf:
        pred = hf["reconstruction"][:]

    with h5py.File(args.ground_truth, "r") as hf:
        # Try various target keys
        for key in ["reconstruction_rss", "reconstruction_esc", "target"]:
            if key in hf:
                target = hf[key][:]
                break
        else:
            # Fall back to zero-filled reconstruction from kspace
            kspace = hf["kspace"][:]
            from src.data.kspace_transforms import ifft2c, root_sum_of_squares
            k = torch.tensor(kspace, dtype=torch.complex64)
            if k.dim() == 4:  # Multi-coil
                img = ifft2c(k)
                target = root_sum_of_squares(img.abs(), dim=1).numpy()
            else:
                target = torch.abs(ifft2c(k)).numpy()

    evaluator = MRIEvaluator(device="cpu", compute_fid=False)

    for i in range(len(pred)):
        p = torch.tensor(pred[i]).unsqueeze(0)
        t = torch.tensor(target[i]).unsqueeze(0)
        # Normalize
        max_val = t.max() + 1e-8
        evaluator.update(p / max_val, t / max_val)

    results = evaluator.compute()
    evaluator.print_summary(results)
    return results


def main():
    args = parse_args()

    if args.method in ("pc", "em", "ddim"):
        if args.checkpoint is None:
            print("Error: --checkpoint required for diffusion methods")
            sys.exit(1)
        reconstruct_diffusion(args)

    elif args.method == "cs":
        if args.input is None:
            print("Error: --input required for CS reconstruction")
            sys.exit(1)
        reconstruct_cs(args)

    elif args.method == "unet":
        if args.checkpoint is None:
            print("Error: --checkpoint required for U-Net reconstruction")
            sys.exit(1)
        print("U-Net reconstruction not implemented in this script.")
        print("Use scripts/evaluate.py with --method unet instead.")
        sys.exit(1)

    # Optional metric computation
    if args.compute_metrics and args.ground_truth:
        compute_metrics_comparison(args)


if __name__ == "__main__":
    main()
