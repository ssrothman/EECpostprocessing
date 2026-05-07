#!/usr/bin/env python3

"""Load an UnfoldedHistogram, profile/condition nuisances, and save a Histogram.

The input UnfoldedHistogram is loaded from disk, converted to a basic Histogram
with a CLI-configured NuisanceTreatment, and written back to disk at the
requested output location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fasteigenpy as eigen
import numpy as np

from unfolding.histogram import Histogram, UnfoldedHistogram
from unfolding.specs import NuisanceTreatment

def main() -> int:
	parser = argparse.ArgumentParser(
		description="Load an UnfoldedHistogram from disk, build a basic Histogram, and save it."
	)
	parser.add_argument(
		"--input",
		default="unfolded",
		help="Directory containing the UnfoldedHistogram files (default: current directory)",
		type=str
	)
	parser.add_argument(
		"--output",
		required=True,
		help="Directory where the resulting Histogram will be written",
		type=str
	)
	parser.add_argument(
		"--profile",
		nargs="*",
		default=[],
		help="Nuisance indices to profile out",
		type=int
	)
	parser.add_argument(
		"--fix",
		nargs="*",
		default=[],
		help="Nuisance indices to fix",
		type=int
	)
	parser.add_argument(
		"--fixvals",
		nargs="*",
		default=[],
		help="Values to use for fixed nuisances, in the same order as --fix",
		type=float
	)
	parser.add_argument(
		"--num",
		type=int,
		required=True,
		help="Total number of nuisance parameters",
	)

	args = parser.parse_args()
	input_dir = Path(args.input)
	output_dir = Path(args.output)

	if not input_dir.exists():
		print(f"Error: input path does not exist: {input_dir}", file=sys.stderr)
		return 1
	if not input_dir.is_dir():
		print(f"Error: input path is not a directory: {input_dir}", file=sys.stderr)
		return 1

	if len(args.fixvals) != len(args.fix):
		print("Error: --fixvals must have the same number of entries as --fix", file=sys.stderr)
		return 1

	nuisance_treatment = NuisanceTreatment(
		profile=args.profile,
		fix=args.fix,
		fixvals=args.fixvals,
		num=args.num,
	)

	unfolded_hist = UnfoldedHistogram.from_disk(str(input_dir))

	unfolded_hist.compute_invhess()
	hist = unfolded_hist.to_basic_histogram(nuisance_treatment)

	hist.dump_to_disk(str(output_dir))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
