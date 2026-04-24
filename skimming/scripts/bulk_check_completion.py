#!/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import subprocess
from tqdm import tqdm

def check_one_workspace(wspath: str) -> tuple[str, str, int]:
    with open(os.path.join(wspath, 'config.json'), 'r') as f:
        wsconfig = json.load(f)

    wstables = wsconfig['tables']
    outpath = wsconfig['output_path'].split('/')
    if len(outpath) != 4:
        raise ValueError(f"Could not parse output path: {wsconfig['output_path']}")

    configsuite = outpath[0]
    runtag = outpath[1]
    dataset = outpath[2]
    objsyst = outpath[3]

    location = wsconfig['output_location']
    

    targetfilespath = os.path.join(wspath, 'target_files.txt')

    cmd = (
        'check_completion.py'
        + ' ' + runtag
        + ' ' + dataset
        + ' --objsyst ' + objsyst
        + ' --location ' + location
        + ' --configsuite ' + configsuite
        + ' --tables ' + ' '.join(wstables)
        + ' --target-files-from-file ' + targetfilespath
        + ' --strict'
    )
    
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return wspath, output.stdout, output.returncode

def main() -> int:
    parser = argparse.ArgumentParser(description="Check completion for all workspaces in a given path.")
    parser.add_argument('path', type=str, help='Path to check for completion')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of worker processes')
    args = parser.parse_args()

    all_subpaths = os.listdir(args.path)

    to_do = []
    for subpath in all_subpaths:
        full_subpath = os.path.join(args.path, subpath)
        if not os.path.isdir(full_subpath):
            continue

        if not os.path.exists(os.path.join(full_subpath, 'target_files.txt')):
            continue

        if not os.path.exists(os.path.join(full_subpath, 'skimscript.py')):
            continue

        to_do.append(full_subpath)

    failures = []
    incomplete_count = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(check_one_workspace, subpath): subpath for subpath in to_do}
        progress = tqdm(as_completed(futures), total=len(futures))
        for future in progress:
            subpath, stdout, exitcode = future.result()
            if exitcode != 0:
                incomplete_count += 1
                failures.append((subpath, stdout))
            progress.set_postfix(incomplete=incomplete_count)

    if failures:
        print("Incomplete workspaces:")
        for subpath, stdout in failures:
            print(f"Workspace {subpath} is not complete.")
            print("Output was:")
            print(stdout)
            print("\n")

    return 0

if __name__ == "__main__":
    exit(main())