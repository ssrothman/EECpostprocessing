#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from typing import Sequence

from skimming.tables.expand_tables import table_names


def infer_workspace_metadata(workspace: str) -> tuple[dict[str, str], Sequence[str]]:
    config_path = os.path.join(workspace, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Workspace is missing required file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    output_location = config.get("output_location")
    if not isinstance(output_location, str) or output_location.strip() == "":
        raise RuntimeError("Workspace config.json is missing 'output_location'.")

    output_path = config.get("output_path")
    if not isinstance(output_path, str) or output_path.strip() == "":
        raise RuntimeError("Workspace config.json is missing 'output_path'.")

    parts = output_path.split("/")
    if len(parts) < 4:
        raise RuntimeError(
            "Workspace config.json has malformed 'output_path'; expected '<configsuite>/<runtag>/<dataset>/<objsyst>'."
        )

    config_suite, runtag, dataset, objsyst = parts[0], parts[1], parts[2], parts[3]

    tables = config.get("tables", [])
    if not isinstance(tables, list) or len(tables) == 0:
        raise RuntimeError("Workspace config.json is missing non-empty 'tables'.")

    return (
        {
            "runtag": runtag,
            "dataset": dataset,
            "objsyst": objsyst,
            "location": output_location,
            "config_suite": config_suite,
        },
        table_names(tables)
    )


def choose_next_suffix(workspace: str, scheduler: str) -> int:
    """Choose next suffix for missing files, checking for existing artifacts from the given scheduler."""
    idx = 1
    while True:
        target_files_missing = os.path.join(workspace, f"target_files_missing_{idx}.txt")
        skimscript_missing = os.path.join(workspace, f"skimscript_missing_{idx}.py")
        submit_missing = os.path.join(workspace, f"submit_{scheduler}_missing_{idx}")
        # Also check for alternative extension (.sh for slurm/local, .sub for condor)
        ext = ".sub" if scheduler == "condor" else ".sh"
        submit_missing_ext = os.path.join(workspace, f"submit_{scheduler}_missing_{idx}{ext}")
        condor_exec_missing = os.path.join(workspace, f"condor_exec_missing_{idx}.sh")
        if not any(
            os.path.exists(p)
            for p in [target_files_missing, skimscript_missing, submit_missing, submit_missing_ext, condor_exec_missing]
        ):
            return idx
        idx += 1


def choose_temp_path(workspace: str, prefix: str) -> str:
    idx = 1
    while True:
        candidate = os.path.join(workspace, f".{prefix}_{idx}.txt")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def run_check_missing(
    workspace: str,
    metadata: dict[str, str],
    tables: Sequence[str],
    check_j: int,
    missing_path: str,
    glitched_path: str,
    target_files_from_workspace: bool,
) -> list[str]:
    check_script = os.path.join(os.path.dirname(__file__), "check_missing_files.py")
    cmd = [
        sys.executable,
        check_script,
        metadata["runtag"],
        metadata["dataset"],
        metadata["objsyst"],
        metadata["location"],
        metadata["config_suite"],
        '--tables', *tables,
        "-j",
        str(max(1, check_j)),
        "--write-missing",
        missing_path,
        "--write-glitched",
        glitched_path,
    ]

    if target_files_from_workspace:
        cmd.extend(["--target-files-txt", os.path.join(workspace, "target_files.txt")])

    result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError("check_missing_files.py failed")

    if not os.path.exists(missing_path):
        return []

    with open(missing_path, "r", encoding="utf-8") as f:
        missing = [line.strip() for line in f if line.strip() != ""]

    return sorted(missing)

def make_missing_skimscript(workspace: str, suffix: int) -> str:
    src = os.path.join(workspace, "skimscript.py")
    if not os.path.exists(src):
        raise RuntimeError(f"Workspace is missing required file: {src}")

    dst_name = f"skimscript_missing_{suffix}.py"
    dst = os.path.join(workspace, dst_name)

    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    replaced = content.replace("./target_files.txt", f"./target_files_missing_{suffix}.txt")
    replaced = replaced.replace("target_files.txt", f"target_files_missing_{suffix}.txt")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(replaced)

    return dst_name


def make_missing_target_file(workspace: str, suffix: int, missing_files: list[str]) -> str:
    dst_name = f"target_files_missing_{suffix}.txt"
    dst = os.path.join(workspace, dst_name)
    with open(dst, "w", encoding="utf-8") as f:
        for tf in sorted(missing_files):
            f.write(tf + "\n")
    return dst_name


def make_missing_slurm_submit(
    workspace: str,
    name: str,
    suffix: int,
    files_per_job: int,
    nfiles: int,
    skimscript_name: str,
    mem: str,
) -> str:
    """Create SLURM submit script for missing files."""
    template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "slurm_template.sh")
    template = os.path.abspath(template)
    if not os.path.exists(template):
        raise RuntimeError(f"Missing slurm template file: {template}")

    with open(template, "r", encoding="utf-8") as f:
        content = f.read()

    njobs = nfiles // files_per_job
    content = content.replace("NAME", name)
    content = content.replace("NJOBS", str(njobs))
    content = content.replace("MEM", mem)
    content = content.replace("WORKINGDIR", workspace)
    content = content.replace("FILES_PER_JOB", str(files_per_job))
    content = content.replace("NFILES", str(nfiles))
    content = content.replace("python skimscript.py $index", f"python {skimscript_name} $index")

    submit_name = f"submit_slurm_missing_{suffix}.sh"
    submit_path = os.path.join(workspace, submit_name)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Mirror behavior of template scripts by marking executable.
    st = os.stat(submit_path)
    os.chmod(submit_path, st.st_mode | 0o111)

    return submit_name


def make_missing_condor_submit(
    workspace: str,
    name: str,
    suffix: int,
    files_per_job: int,
    nfiles: int,
    skimscript_name: str,
    mem: str,
) -> tuple[str, str]:
    """Create Condor submit files for missing files.
    
    Returns tuple of (submit_file_name, exec_file_name).
    """
    submit_template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "condor_submit_template.sh")
    exec_template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "condor_exec_template.sh")
    submit_template = os.path.abspath(submit_template)
    exec_template = os.path.abspath(exec_template)
    
    if not os.path.exists(submit_template):
        raise RuntimeError(f"Missing condor submit template file: {submit_template}")
    if not os.path.exists(exec_template):
        raise RuntimeError(f"Missing condor exec template file: {exec_template}")

    target_name = f"target_files_missing_{suffix}.txt"
    exec_name = f"condor_exec_missing_{suffix}.sh"

    # Create submit file
    with open(submit_template, "r", encoding="utf-8") as f:
        submit_content = f.read()

    njobs = nfiles // files_per_job
    submit_content = submit_content.replace("NAME", name)
    submit_content = submit_content.replace("NJOBS", str(njobs))
    submit_content = submit_content.replace("MEM", mem)
    submit_content = submit_content.replace("condor_exec.sh", exec_name)
    submit_content = submit_content.replace("skimscript.py", skimscript_name)
    submit_content = submit_content.replace("target_files.txt", target_name)

    submit_name = f"submit_condor_missing_{suffix}.sub"
    submit_path = os.path.join(workspace, submit_name)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write(submit_content)

    # Create exec file
    with open(exec_template, "r", encoding="utf-8") as f:
        exec_content = f.read()

    exec_content = exec_content.replace("FILES_PER_JOB", str(files_per_job))
    exec_content = exec_content.replace("NFILES", str(nfiles))

    exec_path = os.path.join(workspace, exec_name)

    exec_content = exec_content.replace("python skimscript.py $index", f"python {skimscript_name} $index")

    with open(exec_path, "w", encoding="utf-8") as f:
        f.write(exec_content)

    # Mark exec file as executable
    st = os.stat(exec_path)
    os.chmod(exec_path, st.st_mode | 0o111)

    return submit_name, exec_name


def make_missing_local_submit(
    workspace: str,
    suffix: int,
    nfiles: int,
    skimscript_name: str,
) -> str:
    """Create a local bash script that runs all missing jobs sequentially."""
    submit_name = f"submit_local_missing_{suffix}.sh"
    submit_path = os.path.join(workspace, submit_name)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"for i in $(seq 0 {nfiles - 1}); do",
        f"    python {skimscript_name} \"$i\"",
        "done",
        "",
    ]

    with open(submit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    st = os.stat(submit_path)
    os.chmod(submit_path, st.st_mode | 0o111)

    return submit_name


def make_missing_submit_script(
    workspace: str,
    name: str,
    suffix: int,
    files_per_job: int,
    nfiles: int,
    skimscript_name: str,
    mem: str,
) -> str:
    """Deprecated: Use make_missing_slurm_submit instead."""
    return make_missing_slurm_submit(workspace, name, suffix, files_per_job, nfiles, skimscript_name, mem)


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
    
    # Mutually exclusive scheduler options
    scheduler_group = parser.add_mutually_exclusive_group(required=True)
    scheduler_group.add_argument("--slurm", action="store_true", help="Submit to SLURM")
    scheduler_group.add_argument("--condor", action="store_true", help="Submit to Condor")
    scheduler_group.add_argument("--local", action="store_true", help="Write a local bash loop script")
    
    parser.add_argument("--exec", action="store_true", help="Submit/run the generated script (sbatch, condor_submit, or bash)")
    parser.add_argument(
        "--keep-temp-check-files",
        action="store_true",
        help="Keep temporary missing/glitched check output files created in the workspace",
    )
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

    metadata, tables = infer_workspace_metadata(workspace)

    print("Using metadata from workspace config:")
    print("  Runtag      :", metadata["runtag"])
    print("  Dataset     :", metadata["dataset"])
    print("  Objsyst     :", metadata["objsyst"])
    print("  Location    :", metadata["location"])
    print("  ConfigSuite :", metadata["config_suite"])
    print("  Tables      :", tables)
    print("  Scheduler   :", scheduler.upper())

    missing_tmp = choose_temp_path(workspace, "missing_files_check")
    glitched_tmp = choose_temp_path(workspace, "glitched_files_check")

    try:
        missing_files = run_check_missing(
            workspace=workspace,
            metadata=metadata,
            tables = tables,
            check_j=args.check_j,
            missing_path=missing_tmp,
            glitched_path=glitched_tmp,
            target_files_from_workspace=args.target_files_from_workspace,
        )

        if len(missing_files) == 0:
            print("No missing files found. Nothing to stage.")
            return

        suffix = choose_next_suffix(workspace, scheduler)
        
        # Create directories for scheduler-specific outputs
        if scheduler == "slurm":
            os.makedirs(os.path.join(workspace, "slurm"), exist_ok=True)
        elif scheduler == "condor":
            os.makedirs(os.path.join(workspace, "condor"), exist_ok=True)
        else:  # local
            os.makedirs(os.path.join(workspace, "local"), exist_ok=True)

        target_name = make_missing_target_file(workspace, suffix, missing_files)
        skimscript_name = make_missing_skimscript(workspace, suffix)

        job_name = args.name if args.name is not None else f"{os.path.basename(workspace)}_missing"
        
        # Generate submit scripts based on scheduler
        if scheduler == "slurm":
            submit_name = make_missing_slurm_submit(
                workspace=workspace,
                name=job_name,
                suffix=suffix,
                files_per_job=args.files_per_job,
                nfiles=len(missing_files),
                skimscript_name=skimscript_name,
                mem=args.mem,
            )
            print("Created missing-file SLURM artifacts:")
            print("  ", target_name)
            print("  ", skimscript_name)
            print("  ", submit_name)

            if args.exec:
                cmd = ["sbatch", submit_name]
                output = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
                if output.stdout:
                    print(output.stdout)
                if output.stderr:
                    print(output.stderr)
                if output.returncode != 0:
                    raise RuntimeError("Failed to submit missing-file SLURM jobs")
            else:
                print("Submit with:")
                print("  sbatch %s" % os.path.join(workspace, submit_name))
        
        elif scheduler == "condor":
            submit_name, exec_name = make_missing_condor_submit(
                workspace=workspace,
                name=job_name,
                suffix=suffix,
                files_per_job=args.files_per_job,
                nfiles=len(missing_files),
                skimscript_name=skimscript_name,
                mem=args.mem,
            )
            print("Created missing-file Condor artifacts:")
            print("  ", target_name)
            print("  ", skimscript_name)
            print("  ", submit_name)
            print("  ", exec_name)

            if args.exec:
                cmd = ["condor_submit", submit_name]
                output = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
                if output.stdout:
                    print(output.stdout)
                if output.stderr:
                    print(output.stderr)
                if output.returncode != 0:
                    raise RuntimeError("Failed to submit missing-file Condor jobs")
            else:
                print("Submit with:")
                print("  condor_submit %s" % os.path.join(workspace, submit_name))

        else:  # local
            submit_name = make_missing_local_submit(
                workspace=workspace,
                suffix=suffix,
                nfiles=len(missing_files),
                skimscript_name=skimscript_name,
            )
            print("Created missing-file local artifacts:")
            print("  ", target_name)
            print("  ", skimscript_name)
            print("  ", submit_name)

            if args.exec:
                print("Running local missing-file script...")
                cmd = ["bash", submit_name]
                output = subprocess.run(cmd, cwd=workspace)
                if output.returncode != 0:
                    raise RuntimeError("Failed to execute local missing-file script")
            else:
                print("Run with:")
                print("  bash %s" % os.path.join(workspace, submit_name))

    finally:
        if not args.keep_temp_check_files:
            for p in [missing_tmp, glitched_tmp]:
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    main()
