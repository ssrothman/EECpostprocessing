#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Stage skimming workspaces to Condor/SLURM/local"
)
parser.add_argument("where", type=str, help="Directory containing skimming workspaces")
parser.add_argument(
    "--files-per-job",
    type=int,
    default=1,
    help="Number of input files to process per job (default: 1)",
)
parser.add_argument(
    "--mem",
    type=str,
    default="4gb",
    help="Memory requested for each job (default: 4gb)",
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
    "-j",
    type=int,
    default=1,
    help="Number of parallel workspace operations (default: 1)",
)

args = parser.parse_args()

if args.files_per_job <= 0:
    raise RuntimeError("--files-per-job must be > 0")
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
    # Delegate to stage_to_condor.py or stage_to_slurm.py (or local)
    if scheduler == "condor":
        script_path = os.path.join(os.path.dirname(__file__), "stage_to_condor.py")
    elif scheduler == "local":
        raise RuntimeError("Local execution is not implemented yet") # placeholder until local execution is implemented; see comment in code below about local execution
    else:
        script_path = os.path.join(os.path.dirname(__file__), "stage_to_slurm.py")

    workspace_name = os.path.basename(os.path.abspath(workspace))

    if args.name_prefix.strip() != "":
        job_name = f"{args.name_prefix}{workspace_name}"
    else:
        job_name = f"{workspace_name}"

    # Build command for delegated script
    if scheduler == "condor":
        cmd = [sys.executable, script_path, './', job_name, "--files-per-job", str(args.files_per_job), "--mem", args.mem]
        if args.exec:
            cmd.append("--exec")
    elif scheduler == "local":
        raise RuntimeError("Local execution is not implemented yet") # placeholder until local execution is implemented; the implementation would likely involve generating a bash script that runs the skimming jobs in a loop, and then executing that script if --exec is specified
    else:
        cmd = [sys.executable, script_path, './', job_name, "--files-per-job", str(args.files_per_job), "--mem", args.mem]
        if args.exec:
            cmd.append("--exec")

    # Execute in the workspace so generated files land there
    output = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace)
    return workspace, output.returncode, output.stdout, output.stderr


def main() -> None:
    where = args.where
    subdirs = os.listdir(where)
    workspaces = []
    for sd in sorted(subdirs):
        fullpath = os.path.join(where, sd)
        if args.filter and any(substr not in sd for substr in args.filter):
            continue
        if args.anti_filter and any(substr in sd for substr in args.anti_filter):
            continue
        if is_workspace(fullpath):
            if args.exec and has_existing_submission_dir(fullpath):
                print("Skipping already-submitted workspace %s" % fullpath)
                continue
            workspaces.append(fullpath)

    import random
    random.shuffle(workspaces)

    if args.condor:
        scheduler = "condor"
    elif args.local:
        scheduler = "local"
    else:
        scheduler = "slurm"

    if len(workspaces) == 0:
        print("No skimming workspaces found in %s" % where)
        return

    for ws in workspaces:
        print("Queueing workspace %s" % ws)

    max_workers = max(1, args.j)
    failures = []
    submitted_jobs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_workspace = {executor.submit(run_one_workspace, ws, scheduler): ws for ws in workspaces}
        for future in tqdm(as_completed(future_to_workspace), total=len(future_to_workspace)):
            workspace = future_to_workspace[future]
            wd, returncode, stdout, stderr = future.result()
            # Print any non-empty output
            if stdout.strip():
                print()
                print("Staging output for %s:" % wd)
                print(stdout)
                # Extract job ids if present
                if 'Submitted batch job' in stdout:
                    import re
                    match = re.search(r'Submitted batch job (\d+)', stdout)
                    if match:
                        job_id = match.group(1)
                        submitted_jobs.append(f"SLURM Job {job_id}")
                if 'job(s) submitted to cluster' in stdout:
                    import re
                    match = re.search(r'(\d+) job\(s\) submitted to cluster (\d+)', stdout)
                    if match:
                        num_jobs = match.group(1)
                        cluster_id = match.group(2)
                        submitted_jobs.append(f"Condor Cluster {cluster_id} with {num_jobs} jobs")
            if stderr.strip():
                print(stderr)
            if returncode != 0:
                failures.append(workspace)

    if args.exec:
        print("THE SUBMITTED JOBS WERE:")
        for job in submitted_jobs:
            print(job)

    if len(failures) > 0:
        raise RuntimeError("Staging failed for workspace(s): %s" % ", ".join(failures))

if __name__ == "__main__":
    main()