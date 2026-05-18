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
    parser.add_argument(
        '--pretty', action='store_true', help="Draw the pretty 2D radial plots"
    )
    parser.add_argument(
        '--transform', type=str, choices=['none', 'shapes', 'angular_modulation'], default='none', help="Apply a transformation to the histogram before plotting."
    )
    parser.add_argument(
        '--projected-r-c', action='store_true', help="Also plot the total flux integrated over (r, c)"
    )
    parser.add_argument(
        '--projected-c', action='store_true', help='Also plot radial profile (integrated over c)'
    )
    parser.add_argument(
        '--covmats', action='store_true', help="Also plot the covariance matrices (both raw and transformed to density space)"
    )

    args = parser.parse_args()

    histogram_dir = Path(args.histogram_path).expanduser().resolve()
    if not histogram_dir.is_dir():
        raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    histogram = Histogram.from_disk(str(histogram_dir))
    histogram.plot(output_folder=os.path.join(str(histogram_dir), 'plots'), 
                   prettymatrices=args.pretty, 
                   covmats=args.covmats,
                   transform=args.transform,
                   projected_r_c_1D=args.projected_r_c,
                   projected_c_1D=args.projected_c)

if __name__ == "__main__":
    main()
