import os.path
import json
from typing import Sequence
from skimming.tables.expand_tables import table_names

def infer_workspace_metadata(workspace: str) -> tuple[dict[str, str], Sequence[str]]:
    config_path = os.path.join(workspace, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Workspace is missing required file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_location = config.get("input_location")
    if not isinstance(input_location, str) or input_location.strip() == "":
        raise RuntimeError("Workspace config.json is missing 'input_location'.")

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
            'input_location': input_location,
            "config_suite": config_suite,
        },
        table_names(tables)
    )

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