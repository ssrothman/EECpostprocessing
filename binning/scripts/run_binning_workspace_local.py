#!/usr/bin/env python3

import argparse
import os

from binning.scaleout.common import find_missing_file

parser = argparse.ArgumentParser(description="Run all commands in a binning workspace locally")
parser.add_argument("where", type=str, help="Workspace directory")
parser.add_argument("-j", type=int, default=1, help="Number of local workers")
parser.add_argument("--fail-fast", action="store_true", help="Stop early on first failure (sequential mode)")
parser.add_argument("--missing", action="store_true", help="Use highest commands_missing_N.txt instead of commands.txt")
args = parser.parse_args()

commands_file = find_missing_file(args.where, required=args.missing) if args.missing else 'commands.txt'

from binning.scaleout.local import run_workspace_locally

failures = run_workspace_locally(
    working_dir=args.where,
    n_workers=args.j,
    fail_fast=args.fail_fast,
    commands_file=commands_file,
)

if len(failures) > 0:
    print("Failed command indexes:", failures)
    raise SystemExit(1)

print("All commands completed successfully.")
