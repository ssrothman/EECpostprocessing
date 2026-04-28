#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.pylab import f
from tqdm import tqdm

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
    "--name-prefix",
    type=str,
    default="",
    help="Optional prefix for generated job names",
)
parser.add_argument(
    "--exec",
    action="store_true",
    help="Submit generated scripts (sbatch, condor_submit, or bash for local)",
)
parser.add_argument(
    "--keep-temp-check-files",
    action="store_true",
    help="Keep temporary files produced by check_missing_files.py in each workspace",
)
parser.add_argument(
    "--target-files-from-workspace",
    action="store_true",
    help="Tell stage_missing to use each workspace target_files.txt for missing-file checks",
)
parser.add_argument(
    "--only-new",
    action="store_true",
    help="Skip workspaces that already have slurm/ or condor/ subdirectories",
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
if args.mem.strip() == "":
    raise RuntimeError("--mem must not be empty")


def is_workspace(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "config.json"))
        and os.path.exists(os.path.join(path, "skimscript.py"))
        and os.path.exists(os.path.join(path, "target_files.txt"))
    )


def has_existing_submission_dir(path: str) -> bool:
    return any(
        os.path.isdir(os.path.join(path, scheduler_dir))
        for scheduler_dir in ("slurm", "condor")
    )


def run_one_workspace(workspace: str, scheduler: str) -> tuple[str, int, str, str]:
    script_path = os.path.join(os.path.dirname(__file__), "stage_missing.py")
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
        "--mem",
        args.mem,
        "--check-j",
        str(args.check_j),
    ]
    
    # Add scheduler flag
    if scheduler == "condor":
        cmd.append("--condor")
    elif scheduler == "local":
        cmd.append("--local")
    else:
        cmd.append("--slurm")

    if args.exec:
        cmd.append("--exec")
    if args.keep_temp_check_files:
        cmd.append("--keep-temp-check-files")
    if args.target_files_from_workspace:
        cmd.append("--target-files-from-workspace")

    output = subprocess.run(cmd, capture_output=True, text=True)
    return workspace, output.returncode, output.stdout, output.stderr


subdirs = os.listdir(args.where)
workspaces = []
for sd in sorted(subdirs):
    fullpath = os.path.join(args.where, sd)
    if args.filter and any(substr not in sd for substr in args.filter):
        continue
    if args.anti_filter and any(substr in sd for substr in args.anti_filter):
        continue
    if is_workspace(fullpath):
        if args.only_new and has_existing_submission_dir(fullpath):
            print("Skipping already-submitted workspace %s" % fullpath)
            continue
        workspaces.append(fullpath)

# randomize the order of the workspaces to avoid overloading the scheduler with similar jobs at the same time if the input directory is sorted in some way (e.g. by dataset or date)
import random
random.shuffle(workspaces)

# Determine scheduler
if args.condor:
    scheduler = "condor"
elif args.local:
    scheduler = "local"
else:
    scheduler = "slurm"

if len(workspaces) == 0:
    print("No skimming workspaces found in %s" % args.where)
    exit(0)

for ws in workspaces:
    print("Queueing workspace %s" % ws)

max_workers = max(1, args.j)
failures = []
submitted_jobs = []  # Keep track of submitted job identifiers if needed for further processing
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_workspace = {executor.submit(run_one_workspace, ws, scheduler): ws for ws in workspaces}
    for future in tqdm(as_completed(future_to_workspace), total = len(future_to_workspace)):
        workspace = future_to_workspace[future]
        wd, returncode, stdout, stderr = future.result()
        if 'No missing files found.' in stdout:
            continue
        else:
            print()
            print("Missing-job staging output for %s:" % wd)
            if stdout.strip():
                print(stdout)
            if 'Submitted batch job' in stdout:
                # there should be a line like "Submitted batch job %d" in the output for SLURM
                # use re to extract the job ID if needed for tracking
                import re
                match = re.search(r'Submitted batch job (\d+)', stdout)
                if match:
                    job_id = match.group(1)
                    submitted_jobs.append(f"SLURM Job {job_id}")
                else:
                    print("WARNING: Could not find job ID in SLURM submission output: %s" % stdout)
            if 'job(s) submitted to cluster' in stdout:
                # Condor submission output typically includes a line like "X job(s) submitted to cluster Y."
                import re
                match = re.search(r'(\d+) job\(s\) submitted to cluster (\d+)', stdout)
                if match:
                    num_jobs = match.group(1)
                    cluster_id = match.group(2)
                    submitted_jobs.append(f"Condor Cluster {cluster_id} with {num_jobs} jobs")
                else:
                    print("WARNING: Could not find cluster ID in Condor submission output: %s" % stdout)
            if stderr.strip():
                print(stderr)
            if returncode != 0:
                failures.append(workspace)
            print()

if args.exec:
    print("THE SUBMITTED JOBS WERE:")
    for job in submitted_jobs:
        print(job)

if len(failures) > 0:
    raise RuntimeError("Missing-job staging failed for workspace(s): %s" % ", ".join(failures))
