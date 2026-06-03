#!/usr/bin/env python

import fasteigenpy as eigen

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
    parser.add_argument(
        '--pretty',
        action='store_true',
        help="Draw pretty 2D radial plots"
    )
    parser.add_argument(
        '--transform',
        type=str,
        choices=['none', 'shapes', 'angular_modulation', 'angular-modulation'],
        default='none',
        help="Transform to apply to the data before plotting."
    )
    parser.add_argument(
        '--chi2', action='store_true', help="Whether to compute chi2 values"
    )
    parser.add_argument(
        '--chi2-cut', action='store_true', help="Restrict the chi2 computation to the analysis bins of interest (does nothing if --chi2 is not set)"
    )
    args = parser.parse_args()

    # parse \n in extra-text into actual newline characters
    args.extra_text = args.extra_text.replace('\\n', '\n') if args.extra_text else None

    if args.labels is None:
        # need to strip trailing '/' before taking basename
        args.labels = [os.path.basename(p.rstrip('/')) for p in args.histogram_paths]

    args.transform = args.transform.replace('-', '_')
    args.transform = args.transform.lower()

    histogram_dirs = [Path(p).expanduser().resolve() for p in args.histogram_paths]
    for histogram_dir in histogram_dirs:
        if not histogram_dir.is_dir():
            raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    histograms = [Histogram.from_disk(str(histogram_dir)) for histogram_dir in histogram_dirs]

    if args.chi2:
        for i1 in range(len(histograms)):
            for i2 in range(i1 + 1, len(histograms)):
                chi2_value, chi2_nbins = histograms[i1].chi2(histograms[i2], transform=args.transform, cut=args.chi2_cut)
                if chi2_value > 0.01:
                    print(f"Chi2 between '{args.labels[i1]}' and '{args.labels[i2]}': {chi2_value:.2f} ({chi2_nbins} bins)")
                else:
                    # use scientific notation
                    print(f"Chi2 between '{args.labels[i1]}' and '{args.labels[i2]}': {chi2_value:.2e} ({chi2_nbins} bins)")


    Histogram.compare(
        histograms, 
        output_folder=args.output_folder, 
        extratext=args.extra_text, 
        labels_l=args.labels,
        transform=args.transform,
        pretty=args.pretty
    )


if __name__ == "__main__":
    main()
