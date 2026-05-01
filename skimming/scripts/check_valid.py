#!/usr/bin/env python3

"""Check validity of parquet outputs for a skimming workspace.

Reads workspace `config.json` to locate the output directory, then tries
to build a PyArrow dataset. If that fails, scans individual parquet files
and reports those that raise errors when inspected by PyArrow.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Sequence
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from general.fslookup.location_lookup import location_lookup
from skimming.tables.expand_tables import one_table_name, table_names


def infer_output_path(workspace: str) -> tuple[Any, str, Sequence[str]]:
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

    tables = table_names(config['tables'])

    fs, basepath = location_lookup(output_location)
    full_output_path = os.path.join(basepath, output_path)
    return fs, full_output_path, tables

def try_build_dataset(fs : Any, path: str, rm: bool = False, j: int = 1) -> List[str]:
    try:
        dset = ds.dataset(path, filesystem=fs, format="parquet")
        # force a scan of fragments to provoke errors
        fragments = list(dset.get_fragments())
    except Exception as e:
        error_str = str(e)
        # Check if this is a corrupted file error
        if "the file is corrupted" in error_str:
            if rm:
                # Try to extract the file path from the error message
                match = re.search(r"'([^']+\.parquet)'", error_str)
                if match:
                    bad_file = match.group(1)
                    print(f"Removing corrupted file: {bad_file}")
                    try:
                        fs.rm(bad_file)
                        print(f"  -> removed {bad_file}")
                        # Recursively retry building the dataset
                        return try_build_dataset(fs, path, rm=rm)
                    except Exception as rm_e:
                        print(f"  -> failed to remove {bad_file}: {repr(rm_e)}")
                        raise
        
        print("ERROR: building dataset failed:")
        print(repr(e))
        raise

    def check_fragment(fragment_path: str) -> None:
        # Read metadata to check structural validity.
        pf = pq.ParquetFile(fragment_path, filesystem=fs)
        _ = pf.metadata

        # Try to read first row group to detect page-level corruption.
        # Metadata can be valid even if data pages are corrupted.
        #_ = pf.read_row_group(0)
        _ = pq.read_table(fragment_path, filesystem=fs)

    fails = []
    iterator = tqdm(fragments, desc="Checking parquet files", unit="file", total=len(fragments))

    if len(fragments) == 0:
        return fails

    if j > 1:
        with ThreadPoolExecutor(max_workers=j) as executor:
            future_to_fragment = {executor.submit(check_fragment, frag.path): frag.path for frag in fragments}
            for future in as_completed(future_to_fragment):
                frag_path = future_to_fragment[future]
                try:
                    future.result()
                except Exception:
                    fails.append(frag_path)
                iterator.update(1)
                iterator.set_postfix_str(f"Fails: {len(fails)}")
    else:
        for frag in iterator:
            try:
                check_fragment(frag.path)
            except Exception:
                fails.append(frag.path)
                iterator.set_postfix_str(f"Fails: {len(fails)}")

    return fails

def main() -> None:
    parser = argparse.ArgumentParser(description="Check parquet validity for a skimming workspace")
    parser.add_argument("workspace", nargs="?", default=".", help="Skimming workspace directory (default: current dir)")
    parser.add_argument("-j", type=int, default=1, help="Parallel workers for parquet file validation (default: 1)")
    parser.add_argument('--rm', action='store_true', help="Remove bad parquet files in addition to just reporting them")
    
    # Stage missing options
    parser.add_argument('--stage-missing', action='store_true', help="After validation, stage missing files using stage_missing.py")
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
    
    # Mutually exclusive scheduler options (only required if --stage-missing is set)
    scheduler_group = parser.add_mutually_exclusive_group()
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

    workspace = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace):
        raise RuntimeError(f"Workspace directory does not exist: {workspace}")
    
    # Validate scheduler options if --stage-missing is set
    if args.stage_missing and not (args.slurm or args.condor or args.local):
        raise RuntimeError("When using --stage-missing, one of --slurm, --condor, or --local must be specified")
    
    if args.files_per_job <= 0:
        raise RuntimeError("--files-per-job must be > 0")

    fs, out_path, tables = infer_output_path(workspace)

    fails = []

    for table in tables:
        if table == 'cutflow' or table == 'count':
            continue
        print(f"Checking table: {table}")
        table_path = os.path.join(out_path, table)
        fails.extend(try_build_dataset(fs, table_path, rm=args.rm, j=args.j))

    if fails:
        print(f"FAILED: The following files failed to build:")
        for f in fails:
            print(f"  - {f}")
            if args.rm:
                try:
                    fs.rm(f)
                    print(f"    -> removed {f}")
                except Exception as e:
                    print(f"    -> failed to remove {f}: {repr(e)}")
    else:
        print(f"OK: All files built successfully.")
        
    if args.stage_missing:
        print("\nStaging missing files...")
        stage_script = os.path.join(os.path.dirname(__file__), "stage_missing.py")
        cmd = [
            sys.executable,
            stage_script,
            workspace,
            "--files-per-job",
            str(args.files_per_job),
            "--mem",
            args.mem,
            "--check-j",
            str(args.check_j),
        ]
        
        # Add scheduler flag
        if args.condor:
            cmd.append("--condor")
        elif args.local:
            cmd.append("--local")
        else:
            cmd.append("--slurm")
        
        if args.exec:
            cmd.append("--exec")
        if args.keep_temp_check_files:
            cmd.append("--keep-temp-check-files")
        if args.target_files_from_workspace:
            cmd.append("--target-files-from-workspace")
        
        result = subprocess.run(cmd, cwd=workspace)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
