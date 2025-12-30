


def stage_via_slurm(working_dir, name):
    import os
    import subprocess
    import shutil

    #fully resolve working_dir
    working_dir = os.path.abspath(working_dir)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), 'templates', 'slurm_template.sh'),
        os.path.join(working_dir, 'submit_slurm.sh')
    )

    with open(os.path.join(working_dir, 'target_files.txt')) as f:
        nfiles = sum(1 for _ in f)

    # customize slurm script
    # by replacing all instances of NAME with `name`
    # and `NJOBS` with nfiles
    # use sed for this

    slurm_script_path = os.path.join(working_dir, 'submit_slurm.sh')
    subprocess.run([
        'sed', '-i',
        f's|NAME|{name}|g; s|NJOBS|{nfiles - 1}|g; s|WORKINGDIR|{working_dir}|g',
        slurm_script_path
    ], check=True)