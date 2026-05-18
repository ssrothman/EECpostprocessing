#!/usr/bin/env python3

"""Build an UnfoldedHistogram from a minimization result and cache its inverse Hessian.

This script:
1. Reads an UnfoldedHistogram from a minimization result directory.
2. Calls compute_invhess().
3. Writes the updated object back to disk in the same location.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fasteigenpy as eigen

from unfolding.histogram import UnfoldedHistogram


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Load an UnfoldedHistogram from a minimization result, compute its inverse Hessian, and save it back."
	)
	parser.add_argument(
		"--unfolded-dir",
		default='unfolded',
		help="Directory in which to save the unfolded histogram",
	)
	parser.add_argument(
		'--minimization-dir', 
		default='minimization',
		help="Directory containing the minimization result files",
    )

	args = parser.parse_args()
	unfdir = Path(args.unfolded_dir)
	minidir = Path(args.minimization_dir)

	if not minidir.exists():
		print(f"Error: path does not exist: {minidir}", file=sys.stderr)
		return 1
	if not minidir.is_dir():
		print(f"Error: path is not a directory: {minidir}", file=sys.stderr)
		return 1

	try:
		unfolded_hist = UnfoldedHistogram.from_minimization_result(str(minidir))
	except Exception as exc:
		print(f"Error loading UnfoldedHistogram from {minidir}: {exc}", file=sys.stderr)
		return 1

	try:
		unfolded_hist.compute_invhess()
	except Exception as exc:
		print(f"Error computing inverse Hessian for {minidir}: {exc}", file=sys.stderr)
		return 1

	try:
		unfolded_hist.dump_to_disk(str(unfdir))
	except Exception as exc:
		print(f"Error writing UnfoldedHistogram to {unfdir}: {exc}", file=sys.stderr)
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
