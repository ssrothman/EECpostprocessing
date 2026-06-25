#!/usr/bin/env python3

import argparse
import os

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find missing skim inputs in an existing workspace and stage them for execution without overwriting workspace files"
    )
    parser.add_argument("where", nargs="?", default=".", help="Skimming workspace directory (default: current directory)")
    parser.add_argument("name", nargs="?", default=None, help="Job name (default: <workspace>_missing)")
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
    parser.add_argument("--check-j", type=int, default=1, help="Parallel workers for check_missing_files.py (default: 1)")
    parser.add_argument(
        "--target-files-from-workspace",
        action="store_true",
        help="Use workspace target_files.txt instead of dataset file lookup in check_missing_files.py",
    )
    parser.add_argument('--split-by-rows', type=int, default=-1, help="Split the job by the number of rows in the input file")
    
    # Mutually exclusive scheduler options
    scheduler_group = parser.add_mutually_exclusive_group(required=True)
    scheduler_group.add_argument("--slurm", action="store_true", help="Submit to SLURM")
    scheduler_group.add_argument("--condor", action="store_true", help="Submit to Condor")
    scheduler_group.add_argument("--local", action="store_true", help="Write a local bash loop script")
    
    parser.add_argument("--exec", action="store_true", help="Submit/run the generated script (sbatch, condor_submit, or bash)")

    args = parser.parse_args()

    workspace = os.path.abspath(args.where)
    if not os.path.isdir(workspace):
        raise RuntimeError(f"Workspace directory does not exist: {workspace}")

    if args.files_per_job <= 0:
        raise RuntimeError("--files-per-job must be > 0")
    if args.mem.strip() == "":
        raise RuntimeError("--mem must not be empty")

    # Determine scheduler (exactly one must be specified due to required mutually_exclusive_group)
    if args.condor:
        scheduler = "condor"
    elif args.local:
        scheduler = "local"
    else:
        scheduler = "slurm"
    
    from skimming.util.check_missing import stage_missing

    stage_missing(
        workspace=workspace,
        scheduler=scheduler,
        files_per_job=args.files_per_job,
        mem=args.mem,
        exec=args.exec,
        check_j=args.check_j,
        split_by_rows=args.split_by_rows
    )

if __name__ == "__main__":
    main()
