#!/usr/bin/env python

import fasteigenpy as eigen

import argparse
from pathlib import Path

from unfolding.histogram import Histogram, UnfoldedHistogram
import numpy as np
import os.path

def unfolded_to_histogram(uh: UnfoldedHistogram) -> Histogram:
    nGen = len(uh.baseline)
    x_gen = uh.x[:nGen]
    values = x_gen * uh.baseline
    invhess_gen = uh.invhess[:nGen, :nGen]
    d = uh.baseline
    covmat = d[:, None] * invhess_gen * d[None, :]
    return Histogram(values, covmat, uh.binning)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load unfolding Histograms and render diagnostic comparison plots."
    )
    parser.add_argument(
        "histogram_paths",
        nargs="+",
        type=str,
        help="Path to saved unfolded histogram directories",
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
    parser.add_argument(
        '--chi2', action='store_true', help="Whether to compute chi2 values"
    )
    args = parser.parse_args()

    args.extra_text = args.extra_text.replace('\\n', '\n') if args.extra_text else None

    if args.labels is None:
        args.labels = [os.path.basename(p.rstrip('/')) for p in args.histogram_paths]

    histogram_dirs = [Path(p).expanduser().resolve() for p in args.histogram_paths]
    for histogram_dir in histogram_dirs:
        if not histogram_dir.is_dir():
            raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    unfolded = [UnfoldedHistogram.from_disk(str(d)) for d in histogram_dirs]
    histograms = [unfolded_to_histogram(uh) for uh in unfolded]

    if args.chi2:
        for i1 in range(len(histograms)):
            for i2 in range(i1 + 1, len(histograms)):
                chi2_value = histograms[i1].chi2(histograms[i2])
                print(f"Chi2 between '{args.labels[i1]}' and '{args.labels[i2]}': {chi2_value:.2g}")

    Histogram.compare(histograms, output_folder=args.output_folder, extratext=args.extra_text, labels_l=args.labels)


if __name__ == "__main__":
    main()
