#!/usr/bin/env python

import argparse
from pathlib import Path

from unfolding.histogram import UnfoldedHistogram
import os.path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a saved unfolding Histogram and render its diagnostic plots."
    )
    parser.add_argument(
        "histogram_path",
        help="Path to a saved histogram directory containing values.npy/cov.npy/invcov.npy/bincfg.json",
    )
    parser.add_argument(
        '--covmats',
        action='store_true',
        help="Whether to also plot the covariance matrices (can be large and slow to render)"
    )
    args = parser.parse_args()

    histogram_dir = Path(args.histogram_path).expanduser().resolve()
    if not histogram_dir.is_dir():
        raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    histogram = UnfoldedHistogram.from_disk(str(histogram_dir))
    histogram.plot(output_folder=os.path.join(str(histogram_dir), 'plots'), covmats=args.covmats)

if __name__ == "__main__":
    main()
