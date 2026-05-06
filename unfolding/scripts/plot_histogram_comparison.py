#!/usr/bin/env python

import argparse
from pathlib import Path

from unfolding.histogram import Histogram
import os.path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load unfolding Histograms and render diagnostic comparison plots."
    )
    parser.add_argument(
        "histogram_paths",
        nargs="+",
        type=str,
        help="Path to saved histogram directories containing values.npy/cov.npy/invcov.npy/bincfg.json",
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        help="Path to the output folder where the plots will be saved.",
        default='./plots'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        type=str,
        help="Labels for the histograms.",
        default=None
    )
    parser.add_argument(
        '--extra-text',
        type=str,
        help="Additional text to include in the plots.",
        default=None
    )
    args = parser.parse_args()

    # parse \n in extra-text into actual newline characters
    args.extra_text = args.extra_text.replace('\\n', '\n') if args.extra_text else None

    if args.labels is None:
        # need to strip trailing '/' before taking basename
        args.labels = [os.path.basename(p.rstrip('/')) for p in args.histogram_paths]

    histogram_dirs = [Path(p).expanduser().resolve() for p in args.histogram_paths]
    for histogram_dir in histogram_dirs:
        if not histogram_dir.is_dir():
            raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    histograms = [Histogram.from_disk(str(histogram_dir)) for histogram_dir in histogram_dirs]
    Histogram.compare(histograms, output_folder=args.output_folder, extratext=args.extra_text, labels_l=args.labels)


if __name__ == "__main__":
    main()
