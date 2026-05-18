#!/usr/bin/env python

import argparse
from pathlib import Path

from unfolding.detectormodel import DetectorModel
from unfolding.loss import Loss
from unfolding.minimizer import Minimizer
from unfolding.histogram import Histogram

import os.path
import numpy as np
import torch

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an unfolding minimization from a saved workspace and store the result."
    )
    parser.add_argument(
        "--reco-path",
        default="reco",
        help="Path to the saved reco histogram directory (default: reco)",
    )
    parser.add_argument(
        "--model-path",
        default="model",
        help="Path to the saved detector model directory (default: detectormodel)",
    )
    parser.add_argument(
        "--baseline-path",
        default="mcgen",
        help="Path to the saved gen-baseline histogram directory (default: mcgen)",
    )
    parser.add_argument(
        "--output-dir",
        default="minimization",
        help="Directory where minimization artifacts will be written (default: minimization)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the minimization on (default: cpu)",
    )
    parser.add_argument(
        "--method",
        default="l-bfgs",
        help="torchmin optimization method to use (default: l-bfgs)",
    )
    parser.add_argument(
        "--cpt-interval",
        type=int,
        default=10,
        help="Write a checkpoint every N iterations (default: 10)",
    )
    parser.add_argument(
        "--negative-penalty",
        type=float,
        default=1e6,
        help="Penalty strength for negative unfolded bins (default: 1e6)",
    )
    parser.add_argument(
        "--continue-from",
        action="store_true",
        help="Continue from an existing minimization directory instead of starting fresh",
    )
    parser.add_argument(
        '--beta0', 
        type=str,
        default='ones',
        help="Initialization for the minimization vector x; can be 'ones', 'random', or a path to a .npy file containing the initial vector (default: ones). Ignored if --continue-from is set."
    )
    parser.add_argument(
        '--theta0', 
        type=str,
        default='zeros',
        help="Initialization for the nuisance parameters; can be 'zeros', 'random', or a path to a .npy file containing the initial vector (default: zeros). Ignored if --continue-from is set."
    )
    parser.add_argument(
        '--gtol',
        type=float,
        default=None,
        help='Gradient tolerance for the minimization (default: None)'
    )
    args = parser.parse_args()

    reco_dir = Path(args.reco_path).expanduser().resolve()
    model_dir = Path(args.model_path).expanduser().resolve()
    baseline_dir = Path(args.baseline_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    reco = Histogram.from_disk(str(reco_dir))
    baseline = Histogram.from_disk(str(baseline_dir))
    model = DetectorModel.from_disk(str(model_dir))

    if baseline.values.shape != (model.nGen,):
        raise ValueError(
            f"Baseline histogram must have shape ({model.nGen},), got {baseline.values.shape}"
        )

    loss = Loss(
        reco=reco,
        genbaseline=baseline.values,
        model=model,
        negativePenalty=args.negative_penalty,
    )

    mincfg = {
        "logpath": str(output_dir),
        "method": args.method,
        "cpt_interval": args.cpt_interval,
        "cpt_start": 0,
        "method_options": {},
    }
    if args.gtol is not None:
        mincfg['method_options']['gtol'] = args.gtol

    if args.continue_from and output_dir.exists():
        minimizer, x0 = Minimizer.continue_from(str(output_dir))
    else:
        minimizer = Minimizer(mincfg)
        if args.beta0 == 'ones':
            beta0 = np.ones(model.nGen)
        elif args.beta0 == 'random':
            beta0 = np.random.rand(model.nGen)
        else:
            beta0_path = Path(args.beta0).expanduser().resolve()
            if not beta0_path.is_file():
                raise FileNotFoundError(f"beta0 file not found: {beta0_path}")
            beta0 = np.load(beta0_path)
        
        if args.theta0 == 'zeros':
            theta0 = np.zeros(model.nSyst)
        elif args.theta0 == 'random':
            theta0 = np.random.rand(model.nSyst)
        else:
            theta0_path = Path(args.theta0).expanduser().resolve()
            if not theta0_path.is_file():
                raise FileNotFoundError(f"theta0 file not found: {theta0_path}")
            theta0 = np.load(theta0_path)

        x0 = np.concatenate([beta0, theta0])
        x0 = torch.from_numpy(x0).to(args.device)

    result = minimizer(
        loss,
        x0=x0,
        device=args.device
    )

    if result is None:
        raise RuntimeError("Unfolding minimization failed; no result was produced.")

    print(f"Saved minimization artifacts to: {output_dir}")

if __name__ == "__main__":
    main()