#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.pylab import f
from tqdm import tqdm

from skimming.util.check_missing import stage_all_missing

parser = argparse.ArgumentParser(
    description="Find and stage missing skimming jobs for all workspaces in a directory"
)
parser.add_argument("where", type=str, help="Directory containing skimming workspaces")
parser.add_argument(
    "--files-per-job",
    type=int,
    default=1,
    help="Number of missing input files per task (default: 1)",
)
parser.add_argument(
    "--mem",
    type=str,
    default="4gb",
    help="Memory requested for each resubmission job (default: 4gb)",
)
parser.add_argument(
    "--check-j",
    type=int,
    default=1,
    help="Parallel workers passed to check_missing_files.py in each workspace (default: 1)",
)
parser.add_argument(
    "--filter",
    nargs="+",
    default=None,
    metavar="SUBSTR",
    help="One or more substrings; each must be present in the workspace folder name",
)
parser.add_argument(
    "--anti-filter",
    nargs="+",
    default=None,
    metavar="SUBSTR",
    help="One or more substrings; workspaces containing any of these are excluded",
)

# Mutually exclusive scheduler options
scheduler_group = parser.add_mutually_exclusive_group(required=True)
scheduler_group.add_argument("--slurm", action="store_true", help="Submit to SLURM")
scheduler_group.add_argument("--condor", action="store_true", help="Submit to Condor")
scheduler_group.add_argument("--local", action="store_true", help="Write and run local bash loop scripts")

parser.add_argument(
    "--exec",
    action="store_true",
    help="Submit generated scripts (sbatch, condor_submit, or bash for local)",
)
parser.add_argument(
    "--only-new",
    action="store_true",
    help="Skip workspaces that already have slurm/ or condor/ subdirectories",
)
args = parser.parse_args()

if args.files_per_job <= 0:
    raise RuntimeError("--files-per-job must be > 0")
if args.check_j <= 0:
    raise RuntimeError("--check-j must be > 0")
if args.mem.strip() == "":
    raise RuntimeError("--mem must not be empty")

# Determine scheduler
if args.condor:
    scheduler = "condor"
elif args.local:
    scheduler = "local"
else:
    scheduler = "slurm"

failures, nmissing = stage_all_missing(
    workspace_dir=args.where,
    scheduler=scheduler,
    files_per_job=args.files_per_job,
    mem=args.mem,
    check_j=args.check_j,
    only_new=args.only_new,
    filter=args.filter,
    anti_filter=args.anti_filter,
    exec=args.exec,
)
if failures:
    print("Failed to stage missing files for the following %d workspaces:" % len(failures))
    for f in failures:
        print("  ", f)

if nmissing:
    print("Staged missing files for the following workspaces:")
    for ws, missing in nmissing:
        print("  %s: %d missing files" % (ws, missing))
else:
    print("No missing files found in any workspaces")