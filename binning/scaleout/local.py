from __future__ import annotations
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def _read_commands(working_dir: str, commands_file: str = "commands.txt") -> list[str]:
    commands_path = os.path.join(working_dir, commands_file)
    with open(commands_path) as f:
        return [line.strip() for line in f if line.strip()]

def _run_command(working_dir: str, index: int, command: str) -> tuple[int, int]:
    logs_dir = os.path.join(working_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"command_{index}.log")

    print(f"[{index}] starting -> {log_path}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print(f"[{index}] done with exit code {result.returncode}")
    return index, result.returncode


def run_workspace_locally(
    working_dir: str,
    n_workers: int = 1,
    fail_fast: bool = False,
    commands_file: str = "commands.txt",
) -> list[int]:
    commands = _read_commands(working_dir, commands_file)

    if len(commands) == 0:
        print("No commands to run.")
        return []

    failures: list[int] = []

    if n_workers <= 1:
        for i, command in enumerate(commands):
            _, code = _run_command(working_dir, i, command)
            if code != 0:
                failures.append(i)
                if fail_fast:
                    break
        return failures

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_command, working_dir, i, command): i
            for i, command in enumerate(commands)
        }
        for future in as_completed(futures):
            i, code = future.result()
            if code != 0:
                failures.append(i)

    failures.sort()
    return failures
