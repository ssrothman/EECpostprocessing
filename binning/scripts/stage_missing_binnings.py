#!/usr/bin/env python3
"""
Scan a binning workspace's `commands.txt`, run each binning command in "check" mode
to determine which tables still need to be run, and write a `commands_missing_<N>.txt`
containing the commands (with only the missing tables) that still need to be executed.

Usage: run this in a binning workspace (the directory containing `commands.txt`).
Writes `commands_missing_<N>.txt` when there are missing tables and exits with code 1.
If no missing tables, exits with code 0.

Runs commands in parallel with `-j`.
"""

import argparse
import glob
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from general.fslookup.hist_lookup import get_hist_path

def read_commands(path):
    with open(path, 'r') as f:
        lines = [l.rstrip('\n') for l in f]
    # keep non-empty, non-comment lines
    return [l for l in lines if l and not l.lstrip().startswith('#')]

def run_one_check(cmd:str, validate:bool):
    # parse the command, and identify all the arguments
    # and then rather than execute bin.py, just run the underlying logic
    tokens = shlex.split(cmd)
    runtag = tokens[1]
    dataset = tokens[2]
    objsyst = tokens[3]
    wtsyst = tokens[4]
    # then we have to search for --tables
    if '--tables' in tokens:
        idx = tokens.index('--tables')
        tables = []
        for t in tokens[idx+1:]:
            if t.startswith('-'):
                break
            tables.append(t)
    else:
        tables = [tokens[5]]

    #and similarly search for --location, --config-suite, --statN (optinoal) --statK (optional)
    if '--location' in tokens:
        idx = tokens.index('--location')
        location = tokens[idx+1]
    else:
        raise ValueError("Command does not contain --location: %s" % cmd)
    if '--config-suite' in tokens:
        idx = tokens.index('--config-suite')
        config_suite = tokens[idx+1]
    else:        
        raise ValueError("Command does not contain --config-suite: %s" % cmd)
    # statN and statK are optional, so we just look for them if they exist
    if '--statN' in tokens:
        idx = tokens.index('--statN')
        statN = int(tokens[idx+1])
    else:
        statN = -1
    if '--statK' in tokens:
        idx = tokens.index('--statK')
        statK = int(tokens[idx+1])
    else:
        statK = -1
    
    if '--cov' in tokens:
        cov = True
    else:
        cov = False

    if '--reweighted-suffix' in tokens:
        idx = tokens.index('--reweighted-suffix')
        reweighted_suffix = tokens[idx+1]
    else:
        reweighted_suffix = None

    # now we can just lookup the path to the binned .npy file
    missing_tables = []
    
    for table in tables:
        fs, path = get_hist_path(
            location,
            config_suite,
            runtag,
            dataset,
            objsyst,
            wtsyst,
            table,
            cov,
            statN,
            statK,
            reweighted_suffix
        )
        if fs.exists(path):
            if validate:
                try:
                    with fs.open(path, 'rb') as f:
                        _ = np.load(f)
                except Exception:
                    missing_tables.append(table)
        else:
            missing_tables.append(table)

    # then we parse the missing tables back into a new command
    if missing_tables:
        new_cmd = rebuild_command_with_missing(cmd, missing_tables)
        return (new_cmd, missing_tables)
    else:
        return (None, None)
    
def find_script_index(tokens):
    for i, t in enumerate(tokens):
        bn = os.path.basename(t)
        if bn == 'bin.py' or bn.endswith('bin.py'):
            return i
    return None

def rebuild_command_with_missing(original_cmd, missing_tables):
    # Keep original command as much as possible, but replace the table list
    # If original used --tables, replace that list. Otherwise, replace positional table
    toks = shlex.split(original_cmd)
    if '--tables' in toks:
        idx = toks.index('--tables')
        # find end of table list
        j = idx + 1
        while j < len(toks) and not toks[j].startswith('-'):
            j += 1
        new_toks = toks[:idx+1] + missing_tables + toks[j:]
        return shlex.join(new_toks) if hasattr(shlex, 'join') else ' '.join(shlex.quote(t) for t in new_toks)
    else:
        # try to detect positional table (5th non-option token after the script)
        script_idx = find_script_index(toks)
        if script_idx is not None:
            # gather indices of non-option tokens after script
            nonopt_indices = []
            for i in range(script_idx + 1, len(toks)):
                if not toks[i].startswith('-'):
                    nonopt_indices.append(i)
            if len(nonopt_indices) >= 5:
                # the 5th non-option token is the positional `table`
                table_idx = nonopt_indices[4]
                # remove that token and insert --tables missing_tables at that position
                new_toks = toks[:table_idx] + toks[table_idx+1:]
                insert_at = table_idx
                new_toks = new_toks[:insert_at] + ['--tables'] + missing_tables + new_toks[insert_at:]
                return shlex.join(new_toks) if hasattr(shlex, 'join') else ' '.join(shlex.quote(t) for t in new_toks)
            else:
                # fallback: append --tables missing_tables to the end
                new_toks = toks + ['--tables'] + missing_tables
                return shlex.join(new_toks) if hasattr(shlex, 'join') else ' '.join(shlex.quote(t) for t in new_toks)

def next_missing_filename():
    existing = glob.glob('commands_missing_*.txt')
    nums = []
    for e in existing:
        base = os.path.basename(e)
        try:
            num = int(base.replace('commands_missing_', '').replace('.txt', ''))
            nums.append(num)
        except Exception:
            continue
    n = max(nums) + 1 if nums else 0
    return f'commands_missing_{n}.txt'


def check_one_command(cmd, validate=False):
    """Check a single command for missing tables."""
    new_commands, missing_tables = run_one_check(cmd, validate)
    return (new_commands, missing_tables)


def main():
    parser = argparse.ArgumentParser(description='Stage missing binnings from commands.txt')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Parallel jobs (default: CPU count)')
    parser.add_argument('--commands-file', type=str, default='commands.txt', help='Commands file to read')
    parser.add_argument('--validate-existing', action='store_true', help='Validate existing tables and report any that are corrupted')

    args = parser.parse_args()

    if not os.path.exists(args.commands_file):
        print(f"No {args.commands_file} found in current directory {os.getcwd()}")
        raise SystemExit(2)

    cmds = read_commands(args.commands_file)
    if not cmds:
        print('No commands found in', args.commands_file)
        return

    jobs = args.jobs or (os.cpu_count() or 4)

    new_commands_accu = []  
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(check_one_command, cmd, validate=args.validate_existing): cmd for cmd in cmds}
        iterator = tqdm(as_completed(futs), total=len(futs), desc="Checking commands")
        iterator.set_postfix_str("Missing: 0")
        for fut in iterator:
            original_cmd = futs[fut]
            new_command, missing_tables = fut.result()
            if new_command is not None:
                new_commands_accu.append(new_command)
                iterator.set_postfix_str(f"Missing: {len(new_commands_accu)}")

        iterator.close()

    if not new_commands_accu:
        print('No missing tables found.')
        return
    else:
        fname = next_missing_filename()
        with open(fname, 'w') as f:
            for c in new_commands_accu:
                f.write(c.rstrip() + '\n')

        print("%d missing tables found across %d commands. Staged in %s" % (len(new_commands_accu), len(new_commands_accu), fname))

if __name__ == '__main__':
    main()
