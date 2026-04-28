#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description="Run all commands in a binning workspace locally")
parser.add_argument("where", type=str, help="Workspace directory")
parser.add_argument("-j", type=int, default=1, help="Number of local workers")
parser.add_argument("--fail-fast", action="store_true", help="Stop early on first failure (sequential mode)")
args = parser.parse_args()

from binning.scaleout.local import run_workspace_locally

failures = run_workspace_locally(
    working_dir=args.where,
    n_workers=args.j,
    fail_fast=args.fail_fast,
)

if len(failures) > 0:
    print("Failed command indexes:", failures)
    raise SystemExit(1)

print("All commands completed successfully.")
