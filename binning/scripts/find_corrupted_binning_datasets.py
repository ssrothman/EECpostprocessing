#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


CORRUPTION_PATTERNS = [
    ("snappy", re.compile(r"Corrupt snappy compressed data", re.IGNORECASE)),
    ("thrift_deserialize", re.compile(r"Couldn't deserialize thrift", re.IGNORECASE)),
    ("thrift_invalid_data", re.compile(r"TProtocolException:\s*Invalid data", re.IGNORECASE)),
    ("unknown_type", re.compile(r"don't know what type", re.IGNORECASE)),
]

RUN_COMMAND_RE = re.compile(r"^Running command index\s+(?P<index>\d+):\s+(?P<command>.+)$")
READING_DATASET_RE = re.compile(r"^Reading dataset from\s+(?P<path>.+)$")


@dataclass(frozen=True)
class CorruptionFinding:
    dataset: str
    objsyst: str
    table: str
    job_file: str
    command_index: Optional[int]
    matched_patterns: tuple[str, ...]
    error_snippet: str


def iter_log_pairs(slurm_dir: Path) -> Iterable[tuple[Path, Path]]:
    for out_path in sorted(slurm_dir.glob("*.out")):
        err_path = out_path.with_suffix(".err")
        yield out_path, err_path


def iter_condor_logs(condor_dir: Path) -> Iterable[tuple[Path, Optional[Path]]]:
    # Condor produces three files per job: *.log (submission/meta), *.out (stdout), *.err (stderr).
    # If a job never ran, *.out and *.err will not exist; skip those entries since they can't
    # be used to detect corruption.
    for log in sorted(condor_dir.glob("*.log")):
        stem = log.stem
        out_path = condor_dir / f"{stem}.out"
        err_path = condor_dir / f"{stem}.err"
        if not out_path.exists() and not err_path.exists():
            # job never ran or still pending
            continue

        if out_path.exists():
            yield out_path, (err_path if err_path.exists() else None)
        else:
            # Out missing but err present
            yield err_path, None


def iter_local_logs(local_dir: Path) -> Iterable[tuple[Path, Optional[Path]]]:
    # Local runs typically write combined logs under a logs/ directory as command_*.log
    for log in sorted(local_dir.glob("command_*.log")):
        yield log, None


def parse_command_context(command_line: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        tokens = shlex.split(command_line)
    except ValueError:
        return None, None, None

    script_index = None
    for index, token in enumerate(tokens):
        if token.endswith("bin.py"):
            script_index = index
            break

    if script_index is None:
        return None, None, None

    script_args = tokens[script_index + 1 :]
    if len(script_args) < 4:
        return None, None, None

    dataset = script_args[1]
    objsyst = script_args[2]

    table = None
    if "--tables" in script_args:
        tables_index = script_args.index("--tables")
        if tables_index + 1 < len(script_args):
            table = script_args[tables_index + 1]
    elif len(script_args) >= 5 and not script_args[4].startswith("-"):
        table = script_args[4]

    return dataset, objsyst, table


def extract_context_from_path(path: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    parts = Path(path).parts
    if len(parts) < 2:
        return None, None, None

    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]

    return None, None, None


def detect_corruption(text: str) -> list[str]:
    matched = []
    for label, pattern in CORRUPTION_PATTERNS:
        if pattern.search(text):
            matched.append(label)
    return matched


def summarize_error(text: str, limit: int = 6) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-limit:])


def parse_workspace_logs(workspace: Path, mode: str, dir_name: str) -> list[CorruptionFinding]:
    log_dir = workspace / dir_name
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    findings: list[CorruptionFinding] = []

    if mode == "slurm":
        iterator = iter_log_pairs(log_dir)
    elif mode == "condor":
        iterator = iter_condor_logs(log_dir)
    elif mode == "local":
        iterator = iter_local_logs(log_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for out_path, err_path in iterator:
        out_text = out_path.read_text(errors="replace")
        err_text = ""
        if err_path is not None and err_path.exists():
            err_text = err_path.read_text(errors="replace")
        combined_text = f"{out_text}\n{err_text}"
        matched_patterns = detect_corruption(combined_text)
        if not matched_patterns:
            continue

        command_blocks: list[tuple[Optional[int], str]] = []
        current_command_index = None

        # command lines may appear in combined logs; search combined text
        for line in combined_text.splitlines():
            match = RUN_COMMAND_RE.match(line)
            if match is not None:
                current_command_index = int(match.group("index"))
                command_blocks.append((current_command_index, match.group("command")))

        if not command_blocks:
            dataset = None
            objsyst = None
            table = None
            command_index = None
        else:
            dataset = None
            objsyst = None
            table = None
            command_index = command_blocks[0][0]
            for _, command_line in command_blocks:
                dataset, objsyst, table = parse_command_context(command_line)
                if dataset is not None:
                    break

        if dataset is None or objsyst is None or table is None:
            for line in combined_text.splitlines():
                match = READING_DATASET_RE.match(line)
                if match is not None:
                    dataset, objsyst, table = extract_context_from_path(match.group("path"))
                    if dataset is not None and objsyst is not None and table is not None:
                        break

        if dataset is None:
            dataset = "<unknown dataset>"
        if objsyst is None:
            objsyst = "<unknown objsyst>"
        if table is None:
            table = "<unknown table>"

        findings.append(
            CorruptionFinding(
                dataset=dataset,
                objsyst=objsyst,
                table=table,
                job_file=out_path.name,
                command_index=command_index,
                matched_patterns=tuple(matched_patterns),
                error_snippet=summarize_error(err_text or out_text),
            )
        )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a binning workspace's logs for datasets with corrupted input files."
    )
    parser.add_argument(
        "workspace",
        type=Path,
        help="Path to the binning workspace containing the slurm/ directory",
    )
    parser.add_argument("--slurm", action="store_true", help="Parse SLURM-style logs")
    parser.add_argument("--condor", action="store_true", help="Parse Condor-style logs")
    parser.add_argument("--local", action="store_true", help="Parse local combined logs")

    parser.add_argument(
        "--slurm-dir",
        default="slurm",
        help="Name of the directory containing SLURM logs (default: slurm)",
    )
    parser.add_argument(
        "--condor-dir",
        default="condor",
        help="Name of the directory containing Condor logs (default: condor)",
    )
    parser.add_argument(
        "--local-dir",
        default="logs",
        help="Name of the directory containing local combined logs (default: logs)",
    )
    json_short_group = parser.add_mutually_exclusive_group()
    json_short_group.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human summary",
    )
    json_short_group.add_argument(
        "--short",
        action="store_true",
        help="Print sorted list of (dataset, objsyst, table) triples and exit",
    )
    args = parser.parse_args()

    selected_modes: list[tuple[str, str]] = []
    if args.slurm:
        selected_modes.append(("slurm", args.slurm_dir))
    if args.condor:
        selected_modes.append(("condor", args.condor_dir))
    if args.local:
        selected_modes.append(("local", args.local_dir))

    if not selected_modes:
        raise ValueError("At least one log type must be selected: --slurm, --condor, or --local")

    findings: list[CorruptionFinding] = []
    for mode, dir_name in selected_modes:
        try:
            findings.extend(parse_workspace_logs(args.workspace, mode, dir_name))
        except FileNotFoundError:
            print(f"Warning: log directory not found for mode {mode}: {args.workspace / dir_name}", file=sys.stderr)
            continue

    by_triple: dict[tuple[str, str, str], list[CorruptionFinding]] = defaultdict(list)
    for finding in findings:
        by_triple[(finding.dataset, finding.objsyst, finding.table)].append(finding)

    if args.short:
        for dataset, objsyst, table in sorted(by_triple):
            print(f"({dataset}, {objsyst}, {table})")
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    f"{dataset}::{objsyst}::{table}": [asdict(finding) for finding in triple_findings]
                    for (dataset, objsyst, table), triple_findings in sorted(by_triple.items())
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not by_triple:
        print(f"No corrupted datasets detected in {args.workspace}.")
        return 0

    print(f"Corrupted datasets in {args.workspace}:")
    for dataset, objsyst, table in sorted(by_triple):
        triple_findings = by_triple[(dataset, objsyst, table)]
        print(
            f"- ({dataset}, {objsyst}, {table}) "
            f"({len(triple_findings)} matching log file{'s' if len(triple_findings) != 1 else ''})"
        )
        for finding in triple_findings:
            pattern_text = ", ".join(finding.matched_patterns)
            prefix = f"  - {finding.job_file}"
            if finding.command_index is not None:
                prefix += f" [command index {finding.command_index}]"
            print(f"{prefix}: {pattern_text}")
            if finding.error_snippet:
                print(f"    {finding.error_snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())