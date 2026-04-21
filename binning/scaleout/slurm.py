from __future__ import annotations
def stage_via_slurm(
    working_dir: str,
    name: str,
    commands_per_job: int = 1,
    time: str = "01:00:00",
    mem: str = "4G",
    cpus: int = 1,
):
    import os
    import subprocess
    import shutil

    working_dir = os.path.abspath(working_dir)

    if commands_per_job <= 0:
        raise ValueError("commands_per_job must be > 0")

    with open(os.path.join(working_dir, "commands.txt")) as f:
        ncommands = sum(1 for line in f if line.strip())

    njobs = max(1, (ncommands + commands_per_job - 1) // commands_per_job)
    njobs_minus_one = njobs - 1

    shutil.copyfile(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaleout_templates", "slurm_template.sh"),
        os.path.join(working_dir, "submit_slurm.sh"),
    )

    slurm_script_path = os.path.join(working_dir, "submit_slurm.sh")
    subprocess.run(
        [
            "sed",
            "-i",
            (
                f"s|NAME|{name}|g; "
                f"s|TIME|{time}|g; "
                f"s|MEM|{mem}|g; "
                f"s|CPUS|{cpus}|g; "
                f"s|WORKINGDIR|{working_dir}|g; "
                f"s|COMMANDS_PER_JOB|{commands_per_job}|g; "
                f"s|NCOMMANDS|{ncommands}|g; "
                f"s|NJOBS_MINUS_ONE|{njobs_minus_one}|g"
            ),
            slurm_script_path,
        ],
        check=True,
    )

    os.makedirs(os.path.join(working_dir, "slurm"), exist_ok=True)

    return ncommands
