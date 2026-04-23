#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(
    description="Find and stage missing skimming jobs to SLURM for all workspaces in a directory"
)
parser.add_argument("where", type=str, help="Directory containing skimming workspaces")
parser.add_argument(
    "--files-per-job",
    type=int,
    default=1,
    help="Number of missing input files per SLURM task (default: 1)",
)
parser.add_argument(
    "--check-j",
    type=int,
    default=1,
    help="Parallel workers passed to check_missing_files.py in each workspace (default: 1)",
)
parser.add_argument(
    "--name-prefix",
    type=str,
    default="",
    help="Optional prefix for generated SLURM job names",
)
parser.add_argument(
    "--exec",
    action="store_true",
    help="Submit generated scripts with sbatch",
)
parser.add_argument(
    "--keep-temp-check-files",
    action="store_true",
    help="Keep temporary files produced by check_missing_files.py in each workspace",
)
parser.add_argument(
    "-j",
    type=int,
    default=1,
    help="Number of parallel workspace operations (default: 1)",
)
args = parser.parse_args()

if args.files_per_job <= 0:
    raise RuntimeError("--files-per-job must be > 0")
if args.check_j <= 0:
    raise RuntimeError("--check-j must be > 0")


def is_workspace(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "config.json"))
        and os.path.exists(os.path.join(path, "skimscript.py"))
        and os.path.exists(os.path.join(path, "target_files.txt"))
    )


def run_one_workspace(workspace: str) -> tuple[str, int, str, str]:
    script_path = os.path.join(os.path.dirname(__file__), "run_missing_to_slurm.py")
    workspace_name = os.path.basename(os.path.abspath(workspace))

    if args.name_prefix.strip() != "":
        job_name = f"{args.name_prefix}{workspace_name}_missing"
    else:
        job_name = f"{workspace_name}_missing"

    cmd = [
        sys.executable,
        script_path,
        workspace,
        job_name,
        "--files-per-job",
        str(args.files_per_job),
        "--check-j",
        str(args.check_j),
    ]

    if args.exec:
        cmd.append("--exec")
    if args.keep_temp_check_files:
        cmd.append("--keep-temp-check-files")

    output = subprocess.run(cmd, capture_output=True, text=True)
    return workspace, output.returncode, output.stdout, output.stderr


subdirs = os.listdir(args.where)
workspaces = []
for sd in sorted(subdirs):
    fullpath = os.path.join(args.where, sd)
    if is_workspace(fullpath):
        workspaces.append(fullpath)

if len(workspaces) == 0:
    print("No skimming workspaces found in %s" % args.where)
    exit(0)

for ws in workspaces:
    print("Queueing workspace %s" % ws)

max_workers = max(1, args.j)
failures = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_workspace = {executor.submit(run_one_workspace, ws): ws for ws in workspaces}
    for future in as_completed(future_to_workspace):
        workspace = future_to_workspace[future]
        wd, returncode, stdout, stderr = future.result()
        print("Missing-job staging output for %s:" % wd)
        if stdout.strip():
            print(stdout)
        if returncode != 0:
            if stderr.strip():
                print(stderr)
            failures.append(workspace)

if len(failures) > 0:
    raise RuntimeError("Missing-job staging failed for workspace(s): %s" % ", ".join(failures))
