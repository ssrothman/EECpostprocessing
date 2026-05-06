#!/usr/bin/env python

import argparse
from pathlib import Path

from unfolding.detectormodel import DetectorModel
import os.path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a saved unfolding DetectorModel and render its diagnostic plots."
    )
    parser.add_argument(
        "model_path",
        help="Path to a saved detector model directory",
    )
    parser.add_argument(
        '--detailed', 
        action='store_true', 
        help="Whether to include detailed cuts in the plots (default: False)"
    )
    parser.add_argument(
        '--nuisances',
        action='store_true',
        help="Whether to include nuisance parameters in the plots (default: False)"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")

    model = DetectorModel.from_disk(str(model_dir))
    model.plot(output_folder=os.path.join(str(model_dir), "plots"), detailed=args.detailed, nuisances=args.nuisances)


if __name__ == "__main__":
    main()
