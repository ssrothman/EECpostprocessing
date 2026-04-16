#!/usr/bin/env python

import argparse
from pathlib import Path

from unfolding.histogram import Histogram
import os.path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a saved unfolding Histogram and render its diagnostic plots."
    )
    parser.add_argument(
        "histogram_path",
        help="Path to a saved histogram directory containing values.npy/cov.npy/invcov.npy/bincfg.json",
    )
    args = parser.parse_args()

    histogram_dir = Path(args.histogram_path).expanduser().resolve()
    if not histogram_dir.is_dir():
        raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    histogram = Histogram.from_disk(str(histogram_dir))
    histogram.plot(output_folder=os.path.join(str(histogram_dir), 'plots'))

if __name__ == "__main__":
    main()
