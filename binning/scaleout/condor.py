from __future__ import annotations
def stage_via_condor(
    working_dir: str,
    commands_per_job: int = 1,
    mem: str = "4G",
    cpus: int = 1,
    commands_file: str = "commands.txt",
):
    import os
    import subprocess
    import shutil

    working_dir = os.path.abspath(working_dir)
    job_tag = os.path.basename(working_dir.rstrip(os.sep)) or "binning"

    if commands_per_job <= 0:
        raise ValueError("commands_per_job must be > 0")

    commands_path = os.path.join(working_dir, commands_file)
    with open(commands_path) as f:
        ncommands = sum(1 for line in f if line.strip())

    njobs = max(1, (ncommands + commands_per_job - 1) // commands_per_job)

    shutil.copyfile(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaleout_templates", "condor_submit_template.sh"),
        os.path.join(working_dir, "condor_submit.sh"),
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaleout_templates", "condor_exec_template.sh"),
        os.path.join(working_dir, "condor_exec.sh"),
    )

    condor_submit_path = os.path.join(working_dir, "condor_submit.sh")
    condor_exec_path = os.path.join(working_dir, "condor_exec.sh")

    subprocess.run(
        [
            "sed",
            "-i",
            (
                f"s|NAME|{job_tag}|g; "
                f"s|MEM|{mem}|g; "
                f"s|CPUS|{cpus}|g; "
                f"s|WORKINGDIR|{working_dir}|g; "
                f"s|COMMANDS_PER_JOB|{commands_per_job}|g; "
                f"s|NCOMMANDS|{ncommands}|g; "
                f"s|NJOBS|{njobs}|g; "
                f"s|COMMANDS_FILE|{commands_file}|g"
            ),
            condor_submit_path,
        ],
        check=True,
    )
    subprocess.run(
        [
            "sed",
            "-i",
            (
                f"s|WORKINGDIR|{working_dir}|g; "
                f"s|COMMANDS_PER_JOB|{commands_per_job}|g; "
                f"s|NCOMMANDS|{ncommands}|g; "
                f"s|COMMANDS_FILE|{commands_file}|g"
            ),
            condor_exec_path,
        ],
        check=True,
    )

    subprocess.run(["chmod", "+x", condor_exec_path], check=True)

    os.makedirs(os.path.join(working_dir, "condor"), exist_ok=True)

    return ncommands
