#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def is_workspace(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "config.json"))
        and os.path.exists(os.path.join(path, "skimscript.py"))
        and os.path.exists(os.path.join(path, "target_files.txt"))
    )

def run_one_workspace(workspace: str, rm: bool = False, stage_missing: bool = False, 
                      files_per_job: int = 1, mem: str = "4gb", check_j: int = 1,
                      target_files_from_workspace: bool = False, scheduler: str = "slurm",
                      exec_flag: bool = False, keep_temp_check_files: bool = False) -> tuple[str, int, str, str]:
    script_path = os.path.join(os.path.dirname(__file__), "check_valid.py")
    cmd = [sys.executable, script_path, workspace]
    cmd.extend(["-j", str(check_j)])
    if rm:
        cmd.append("--rm")
    if stage_missing:
        cmd.append("--stage-missing")
        cmd.extend(["--files-per-job", str(files_per_job)])
        cmd.extend(["--mem", mem])
        cmd.extend(["--check-j", str(check_j)])
        cmd.append(f"--{scheduler}")
        if exec_flag:
            cmd.append("--exec")
        if keep_temp_check_files:
            cmd.append("--keep-temp-check-files")
        if target_files_from_workspace:
            cmd.append("--target-files-from-workspace")
    output = subprocess.run(cmd, capture_output=True, text=True)
    return workspace, output.returncode, output.stdout, output.stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run check_valid across all skimming workspaces in a directory"
    )
    parser.add_argument("where", type=str, help="Directory containing skimming workspaces")
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
    parser.add_argument(
        "--rm",
        action="store_true",
        help="Pass through --rm to check_valid.py (if supported)",
    )
    parser.add_argument(
        "--stage-missing",
        action="store_true",
        help="Stage missing files after validation using stage_missing.py",
    )
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
        help="Parallel workers for check_valid.py (-j) and check_missing_files.py when staging (default: 1)",
    )
    parser.add_argument(
        "--target-files-from-workspace",
        action="store_true",
        help="Use workspace target_files.txt instead of dataset file lookup",
    )
    
    # Mutually exclusive scheduler options
    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument("--slurm", action="store_true", help="Submit to SLURM")
    scheduler_group.add_argument("--condor", action="store_true", help="Submit to Condor")
    scheduler_group.add_argument("--local", action="store_true", help="Write a local bash loop script")
    
    parser.add_argument(
        "--exec",
        action="store_true",
        help="Submit/run the generated script (sbatch, condor_submit, or bash)",
    )
    parser.add_argument(
        "--keep-temp-check-files",
        action="store_true",
        help="Keep temporary missing/glitched check output files",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=1,
        help="Number of parallel workspace operations (default: 1)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.where):
        raise RuntimeError(f"Directory does not exist: {args.where}")
    
    # Validate scheduler options if --stage-missing is set
    if args.stage_missing and not (args.slurm or args.condor or args.local):
        raise RuntimeError("When using --stage-missing, one of --slurm, --condor, or --local must be specified")
    
    if args.files_per_job <= 0:
        raise RuntimeError("--files-per-job must be > 0")
    
    # Determine scheduler
    if args.condor:
        scheduler = "condor"
    elif args.local:
        scheduler = "local"
    else:
        scheduler = "slurm"

    subdirs = os.listdir(args.where)
    workspaces = []
    for sd in sorted(subdirs):
        fullpath = os.path.join(args.where, sd)
        if args.filter and any(substr not in sd for substr in args.filter):
            continue
        if args.anti_filter and any(substr in sd for substr in args.anti_filter):
            continue
        if is_workspace(fullpath):
            workspaces.append(fullpath)

    random.shuffle(workspaces)

    if len(workspaces) == 0:
        print("No skimming workspaces found in %s" % args.where)
        return

    max_workers = max(1, args.j)
    failures = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_workspace = {
            executor.submit(
                run_one_workspace, 
                ws, 
                rm=args.rm,
                stage_missing=args.stage_missing,
                files_per_job=args.files_per_job,
                mem=args.mem,
                check_j=args.check_j,
                target_files_from_workspace=args.target_files_from_workspace,
                scheduler=scheduler,
                exec_flag=args.exec,
                keep_temp_check_files=args.keep_temp_check_files,
            ): ws for ws in workspaces
        }
        for future in tqdm(as_completed(future_to_workspace), total=len(future_to_workspace)):
            ws = future_to_workspace[future]
            wd, returncode, stdout, stderr = future.result()
            if stdout and stdout.strip():
                print(f"--- Output for {wd} ---")
                print(stdout)
            if stderr and stderr.strip():
                print(f"--- Stderr for {wd} ---")
                print(stderr)
            if returncode != 0:
                failures.append(wd)
            results.append((wd, returncode))

    print("\nSummary:")
    for wd, rc in results:
        print(f"  {wd}: {'OK' if rc == 0 else 'FAILED'}")

    if failures:
        raise RuntimeError("check_valid failed for workspace(s): %s" % ", ".join(failures))


if __name__ == "__main__":
    main()
