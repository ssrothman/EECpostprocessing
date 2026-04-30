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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def read_commands(path):
    with open(path, 'r') as f:
        lines = [l.rstrip('\n') for l in f]
    # keep non-empty, non-comment lines
    return [l for l in lines if l and not l.lstrip().startswith('#')]


def make_check_command(cmd):
    # remove any --nocheck tokens and add --justcheck and --validate-existing
    toks = shlex.split(cmd)
    toks = [t for t in toks if t != '--nocheck']
    if '--justcheck' not in toks:
        toks.append('--justcheck')
    if '--validate-existing' not in toks:
        toks.append('--validate-existing')
    return toks


def run_check(toks):
    try:
        proc = subprocess.run(toks, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc.returncode, proc.stdout
    except Exception as e:
        return 2, str(e)


def parse_missing_tables(output):
    lines = output.splitlines()
    tables = []
    for i, l in enumerate(lines):
        if l.strip() == 'Tables that need to be run:':
            # collect following non-empty lines
            for ll in lines[i+1:]:
                s = ll.strip()
                if not s:
                    continue
                tables.append(s)
            break
    return tables


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


def check_one_command(cmd):
    """Check a single command for missing tables."""
    toks = make_check_command(cmd)
    rc, out = run_check(toks)
    return (cmd, rc, out)


def main():
    parser = argparse.ArgumentParser(description='Stage missing binnings from commands.txt')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Parallel jobs (default: CPU count)')
    parser.add_argument('--commands-file', type=str, default='commands.txt', help='Commands file to read')

    args = parser.parse_args()

    if not os.path.exists(args.commands_file):
        print(f"No {args.commands_file} found in current directory {os.getcwd()}")
        raise SystemExit(2)

    cmds = read_commands(args.commands_file)
    if not cmds:
        print('No commands found in', args.commands_file)
        return

    jobs = args.jobs or (os.cpu_count() or 4)

    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(check_one_command, cmd): cmd for cmd in cmds}
        iterator = tqdm(as_completed(futs), total=len(futs), desc="Checking commands")
        for fut in iterator:
            original_cmd = futs[fut]
            try:
                cmd, rc, out = fut.result()
            except Exception as e:
                cmd = original_cmd
                rc = 2
                out = str(e)
            results.append((original_cmd, rc, out))

    missing_commands = []
    all_missing_tables = []
    for original_cmd, rc, out in results:
        if rc == 0:
            # no missing tables for this command
            continue
        elif rc == 1:
            tables = parse_missing_tables(out)
            if tables:
                all_missing_tables.extend(tables)
                new_cmd = rebuild_command_with_missing(original_cmd, tables)
                missing_commands.append(new_cmd)
        else:
            # unexpected error; include original command for manual inspection
            print(f"Warning: command returned code {rc}")
            print(f"error message: {out}")
            print(f"original command was {original_cmd}")
            missing_commands.append(original_cmd)

    if not missing_commands:
        print('No missing tables found.')
        return
    else:
        fname = next_missing_filename()
        with open(fname, 'w') as f:
            for c in missing_commands:
                f.write(c.rstrip() + '\n')

        print("%d missing tables found across %d commands. Staged in %s" % (len(all_missing_tables), len(missing_commands), fname))

if __name__ == '__main__':
    main()
