#!/usr/bin/env python

import fasteigenpy as eigen

import argparse
from pathlib import Path
import os.path

from unfolding.histogram import UnfoldedHistogram
from unfolding.specs import NuisanceTreatment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load an UnfoldedHistogram and plot one file per pT bin."
    )
    parser.add_argument(
        "histogram_path",
        help="Path to an unfolded histogram directory (output of build_unfolded_hist.py)",
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help="Output folder for plots (default: <histogram_path>/plots)",
    )
    parser.add_argument(
        '--extra-text',
        type=str,
        default=None,
    )
    args = parser.parse_args()

    histogram_dir = Path(args.histogram_path).expanduser().resolve()
    if not histogram_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {histogram_dir}")

    output_folder = args.output_folder or os.path.join(str(histogram_dir), 'plots')
    extra_text = args.extra_text.replace('\\n', '\n') if args.extra_text else None

    unfolded = UnfoldedHistogram.from_disk(str(histogram_dir))
    unfolded.compute_invhess()
    nuisance_treatment = NuisanceTreatment(profile=[], fix=[], fixvals=[], num=len(unfolded.nuisance_names))
    histogram = unfolded.to_basic_histogram(nuisance_treatment)

    histogram.plot(output_folder=output_folder, extratext=extra_text)


if __name__ == "__main__":
    main()
