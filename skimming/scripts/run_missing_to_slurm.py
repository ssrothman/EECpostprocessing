#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys

from skimming.tables.expand_tables import one_table_name


def infer_workspace_metadata(workspace: str) -> dict[str, str]:
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

    first_table = one_table_name(str(tables[0]))

    return {
        "runtag": runtag,
        "dataset": dataset,
        "objsyst": objsyst,
        "location": output_location,
        "config_suite": config_suite,
        "table": first_table,
    }


def choose_next_suffix(workspace: str) -> int:
    idx = 1
    while True:
        target_files_missing = os.path.join(workspace, f"target_files_missing_{idx}.txt")
        skimscript_missing = os.path.join(workspace, f"skimscript_missing_{idx}.py")
        submit_missing = os.path.join(workspace, f"submit_slurm_missing_{idx}.sh")
        if not any(os.path.exists(p) for p in [target_files_missing, skimscript_missing, submit_missing]):
            return idx
        idx += 1


def choose_temp_path(workspace: str, prefix: str) -> str:
    idx = 1
    while True:
        candidate = os.path.join(workspace, f".{prefix}_{idx}.txt")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def run_check_missing(workspace: str, metadata: dict[str, str], check_j: int, missing_path: str, glitched_path: str) -> list[str]:
    check_script = os.path.join(os.path.dirname(__file__), "check_missing_files.py")
    cmd = [
        sys.executable,
        check_script,
        metadata["runtag"],
        metadata["dataset"],
        metadata["objsyst"],
        metadata["location"],
        metadata["config_suite"],
        metadata["table"],
        "-j",
        str(max(1, check_j)),
        "--write-missing",
        missing_path,
        "--write-glitched",
        glitched_path,
    ]

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


def make_missing_submit_script(workspace: str, name: str, suffix: int, files_per_job: int, nfiles: int, skimscript_name: str) -> str:
    template = os.path.join(os.path.dirname(__file__), "..", "scaleout", "templates", "slurm_template.sh")
    template = os.path.abspath(template)
    if not os.path.exists(template):
        raise RuntimeError(f"Missing slurm template file: {template}")

    with open(template, "r", encoding="utf-8") as f:
        content = f.read()

    njobs = nfiles // files_per_job
    content = content.replace("NAME", name)
    content = content.replace("NJOBS", str(njobs))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find missing skim inputs in an existing workspace and stage them to SLURM without overwriting workspace files"
    )
    parser.add_argument("where", nargs="?", default=".", help="Skimming workspace directory (default: current directory)")
    parser.add_argument("name", nargs="?", default=None, help="SLURM job name (default: <workspace>_missing)")
    parser.add_argument(
        "--files-per-job",
        type=int,
        default=1,
        help="Number of missing input files per SLURM task (default: 1)",
    )
    parser.add_argument("--check-j", type=int, default=1, help="Parallel workers for check_missing_files.py (default: 1)")
    parser.add_argument("--exec", action="store_true", help="Submit the generated script with sbatch")
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

    metadata = infer_workspace_metadata(workspace)

    config_tables_path = os.path.join(workspace, "config.json")
    with open(config_tables_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg.get("tables"), list) and len(cfg["tables"]) > 1:
        print(
            "Multiple tables found in workspace config; using first table: %s -> %s"
            % (cfg["tables"][0], metadata["table"])
        )

    print("Using metadata from workspace config:")
    print("  Runtag      :", metadata["runtag"])
    print("  Dataset     :", metadata["dataset"])
    print("  Objsyst     :", metadata["objsyst"])
    print("  Location    :", metadata["location"])
    print("  ConfigSuite :", metadata["config_suite"])
    print("  Table       :", metadata["table"])

    missing_tmp = choose_temp_path(workspace, "missing_files_check")
    glitched_tmp = choose_temp_path(workspace, "glitched_files_check")

    try:
        missing_files = run_check_missing(
            workspace=workspace,
            metadata=metadata,
            check_j=args.check_j,
            missing_path=missing_tmp,
            glitched_path=glitched_tmp,
        )

        if len(missing_files) == 0:
            print("No missing files found. Nothing to stage.")
            return

        suffix = choose_next_suffix(workspace)
        os.makedirs(os.path.join(workspace, "slurm"), exist_ok=True)

        target_name = make_missing_target_file(workspace, suffix, missing_files)
        skimscript_name = make_missing_skimscript(workspace, suffix)

        job_name = args.name if args.name is not None else f"{os.path.basename(workspace)}_missing"
        submit_name = make_missing_submit_script(
            workspace=workspace,
            name=job_name,
            suffix=suffix,
            files_per_job=args.files_per_job,
            nfiles=len(missing_files),
            skimscript_name=skimscript_name,
        )

        print("Created missing-file rerun artifacts:")
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

    finally:
        if not args.keep_temp_check_files:
            for p in [missing_tmp, glitched_tmp]:
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    main()
