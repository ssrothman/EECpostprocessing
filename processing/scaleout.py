import os
import pickle

from dask.distributed import Client
from dask_jobqueue import SLURMCluster, HTCondorCluster

def setup_htcondor(minjobs, maxjobs, path=None):
    cluster = HTCondorCluster(
            disk = '2GB',
            cores = 1,
            memory = '32GB',
            processes = 1,
            nanny=True,
            job_extra_directives=
                {'+JobFlavor':'espresso',
                 '+AccountingGroup' : 'analysis.srothman',
                 'use_x509userproxy' : 'True',
                 'x509userproxy' : '/home/submit/srothman/myticket',
                 'universe' : 'vanilla',
                 'Requirements' : '( BOSCOCluster =!= "t3serv008.mit.edu" && BOSCOCluster =!= "ce03.cmsaf.mit.edu" && BOSCOCluster =!= "eofe8.mit.edu")',
                 '+DESIRED_Sites' : 'mit_tier3'},
            local_directory='test',
            log_directory='test')
    cluster.adapt(minimum_jobs = minjobs, maximum_jobs = maxjobs)
    client = Client(cluster)
    return cluster, client

def setup_cluster_on_submit(minjobs, maxjobs, path=None):
    import random

    #if path is None:
    uuid = random.getrandbits(64)
    log_directory = 'logs/%016x'%uuid
    #else:
    #    log_directory = os.path.join(path,'logs')

    print("saving logs at",log_directory)

    cluster = SLURMCluster(queue = 'submit,submit-centos07',
                           cores=1,
                           processes=1,
                           memory='4GB',
                           walltime='10:00:00',
                           log_directory=log_directory,
                           scheduler_options={'dashboard_address':":9876"})
    print(cluster.job_script())
    cluster.adapt(
        minimum_jobs=1, maximum_jobs=200,
        worker_key=lambda state: state.address.split(':')[0],
        interval='10s'
    )
    client = Client(cluster)
    return cluster, client

def custom_scale(files, processor_instance, folder):
    N = len(files)

    command_string = 'python run.py $SLURM_ARRAY_TASK_ID'

    with open("%s/submit_array.slurm"%folder, 'w') as f:
        write_slurm_script(f, "scale_array", command_string,
                           mem='50g', array='0-%d%%10'%N, cpus=64)

    with open("%s/files.pkl"%folder, 'wb') as f:
        pickle.dump(files, f)

    with open("%s/processor.pkl"%folder, 'wb') as f:
        pickle.dump(processor_instance, f)

    with open("%s/run.py"%folder, 'w') as f:
        write_run_script(f)

    #for i in range(N):
    #    os.makedirs("%s/%09d"%(folder,i), exist_ok=False)

def write_run_script(f):
    runstring = 'import pickle\n'
    runstring += 'import sys\n'
    runstring += '\n'
    runstring += 'idx = int(sys.argv[1])'
    runstring += '\n'
    runstring += 'with open("files.pkl", "rb") as f:\n'
    runstring += '\tfiles = pickle.load(f)\n'
    runstring += '\n'
    runstring += 'with open("processor.pkl", "rb") as f:\n'
    runstring += '\tprocessor = pickle.load(f)\n'
    runstring += '\n'
    runstring += 'q = processor.locking_merge_on_disk(files[idx], "./hists.pkl")\n'
    runstring += '\n'
    #runstring += 'with open("%09d/hists.pkl"%idx, "wb") as f:\n'
    #runstring += "\tpickle.dump(q, f)\n"
    f.write(runstring)

def write_slurm_script(f, jobname, commandstring,
                       partition='submit,submit-gpu,submit-gpu1080', 
                       time=None, cpus=None, gres=None,
                       mem=None, array=None):
    f.write("#!/bin/bash -l\n")
    if time is not None:
        f.write("#SBATCH --time=%s\n"%time)
    f.write("#SBATCH --ntasks=1\n")
    if cpus is not None:
        f.write("#SBATCH --cpus-per-task=%d\n"%cpus)
    if gres is not None:
        f.write("#SBATCH --gres=%s\n"%gres)
    if mem is not None:
        f.write("#SBATCH --mem=%s\n"%mem)
    f.write("#SBATCH --partition=%s\n"%partition)
    f.write("#SBATCH --mail-type=ALL\n")
    f.write("#SBATCH --mail-user=ssrothman.slurm.span@gmail.com\n")
    f.write("#SBATCH --job-name=\"%s\"\n"%jobname)
    f.write("#SBATCH --output=\"%A_%a.slurmout\"\n")
    if array is not None:
        f.write("#SBATCH --array=%s\n"%array)
    f.write("\n\n")
    f.write(commandstring)
    f.write("\n\n")
