#!/usr/bin/env python3
"""
Script to compute and cache inverse covariance and/or square root decomposition
of a Histogram object.

Usage:
    python compute_invcov.py <path_to_histogram> [--invcov] [--sqrt]

Arguments:
    path_to_histogram: Directory path where the Histogram was saved
    --invcov: Compute and cache the inverse covariance matrix
    --sqrt: Compute and cache the square root decomposition (L and Linv)
"""

import argparse
import sys
from pathlib import Path

import fasteigenpy as eigen
from unfolding.histogram import Histogram


def main():
    parser = argparse.ArgumentParser(
        description="Compute and cache inverse covariance/sqrt for a Histogram object"
    )
    parser.add_argument(
        "histogram_path",
        type=str,
        help="Directory path where the Histogram was saved"
    )
    parser.add_argument(
        "--invcov",
        action="store_true",
        help="Compute and cache the inverse covariance matrix"
    )
    parser.add_argument(
        "--sqrt",
        action="store_true",
        help="Compute and cache the square root decomposition (L and Linv)"
    )

    args = parser.parse_args()

    # Validate that at least one computation was requested
    if not args.invcov and not args.sqrt:
        parser.print_help()
        print("\nError: At least one of --invcov or --sqrt must be specified")
        sys.exit(1)

    histogram_path = Path(args.histogram_path)
    
    # Validate that the path exists
    if not histogram_path.exists():
        print(f"Error: Path does not exist: {histogram_path}")
        sys.exit(1)
    
    if not histogram_path.is_dir():
        print(f"Error: Path is not a directory: {histogram_path}")
        sys.exit(1)

    print(f"Loading Histogram from {histogram_path}...")
    try:
        hist = Histogram.from_disk(str(histogram_path))
    except Exception as e:
        print(f"Error loading Histogram: {e}")
        sys.exit(1)

    # Compute requested quantities
    if args.invcov:
        print("Computing inverse covariance matrix...")
        try:
            hist.compute_invcov()
            print("✓ Inverse covariance computed successfully")
        except Exception as e:
            print(f"Error computing inverse covariance: {e}")
            sys.exit(1)

    if args.sqrt:
        print("Computing square root decomposition...")
        try:
            hist.compute_sqrt()
            print("✓ Square root decomposition computed successfully")
        except Exception as e:
            print(f"Error computing square root: {e}")
            sys.exit(1)

    # Write back to disk
    print(f"Writing Histogram back to {histogram_path}...")
    try:
        hist.dump_to_disk(str(histogram_path))
        print("✓ Histogram saved successfully")
    except Exception as e:
        print(f"Error saving Histogram: {e}")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
