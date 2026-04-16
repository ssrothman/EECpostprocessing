#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np

from unfolding.detectormodel import DetectorModel
from unfolding.forward import forward_hist
from unfolding.histogram import Histogram


def _load_nuisances(path: Path | None, n_syst: int) -> np.ndarray:
    if path is None:
        return np.zeros(n_syst, dtype=float)

    if not path.is_file():
        raise FileNotFoundError(f"Nuisance file does not exist: {path}")

    nuisances = np.load(path)
    if nuisances.ndim != 1:
        raise ValueError(
            f"Nuisances must be a 1D array, got shape {nuisances.shape} from {path}"
        )
    if nuisances.shape[0] != n_syst:
        raise ValueError(
            f"Nuisance length mismatch: model expects {n_syst}, got {nuisances.shape[0]}"
        )

    return nuisances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forward propagation from a saved gen histogram and detector model."
    )
    parser.add_argument(
        "--gen-path",
        default="gen",
        help="Path to saved gen histogram directory (default: gen)",
    )
    parser.add_argument(
        "--model-path",
        default="model",
        help="Path to saved detector model directory (default: model)",
    )
    parser.add_argument(
        "--output",
        default="forward",
        help="Output directory for the forward histogram (default: forward)",
    )
    parser.add_argument(
        "--nuisances",
        default=None,
        help="Optional .npy file containing a 1D nuisance vector; defaults to zeros",
    )
    parser.add_argument(
        "--nboot",
        type=int,
        default=1000,
        help="Number of bootstrap samples for forward covariance (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap sampling (default: 12345)",
    )
    args = parser.parse_args()

    gen_dir = Path(args.gen_path).expanduser().resolve()
    model_dir = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not gen_dir.is_dir():
        raise NotADirectoryError(f"Gen histogram path is not a directory: {gen_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")
    if args.nboot < 2:
        raise ValueError("nboot must be at least 2")

    nuisances_path = None if args.nuisances is None else Path(args.nuisances).expanduser().resolve()

    gen = Histogram.from_disk(str(gen_dir))
    model = DetectorModel.from_disk(str(model_dir))
    nuisances = _load_nuisances(nuisances_path, model.nSyst)

    fwd = forward_hist(gen, nuisances, model, nboot=args.nboot, seed=args.seed)
    fwd.dump_to_disk(str(output_dir))

    print(f"Saved forward histogram to: {output_dir}")


if __name__ == "__main__":
    main()
