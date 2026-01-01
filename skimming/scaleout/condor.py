def stage_via_condor(working_dir, name, files_per_job=1):
    import os
    import subprocess
    import shutil
    
    #fully resolve working_dir
    working_dir = os.path.abspath(working_dir)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), 'templates', 'condor_submit_template.sh'),
        os.path.join(working_dir, 'condor_submit.sh')
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), 'templates', 'condor_exec_template.sh'),
        os.path.join(working_dir, 'condor_exec.sh')
    )

    with open(os.path.join(working_dir, 'target_files.txt')) as f:
        nfiles = sum(1 for _ in f)

    # customize condor submit script
    njobs = nfiles // files_per_job

    condor_sub_path = os.path.join(working_dir, 'condor_submit.sh')
    condor_exec_path = os.path.join(working_dir, 'condor_exec.sh')

    subprocess.run([
        'sed', '-i',
        f's|NAME|{name}|g; s|NJOBS|{njobs}|g; s|WORKINGDIR|{working_dir}|g; s|FILES_PER_JOB|{files_per_job}|g; s|NFILES|{nfiles}|g',
        condor_sub_path
    ], check=True)
    subprocess.run([
        'sed', '-i',
        f's|NAME|{name}|g; s|NJOBS|{njobs}|g; s|WORKINGDIR|{working_dir}|g; s|FILES_PER_JOB|{files_per_job}|g; s|NFILES|{nfiles}|g',
        condor_exec_path
    ], check=True)

    #make directory for logs 
    os.makedirs(os.path.join(working_dir, 'condor'), exist_ok=True)