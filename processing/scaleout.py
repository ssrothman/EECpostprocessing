import pickle

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster, HTCondorCluster

from dask.distributed import WorkerPlugin
from loky import ProcessPoolExecutor

def setup_local_cluster(nworkers):
    cluster = LocalCluster(n_workers=nworkers, 
                           threads_per_worker=1,
                           dashboard_address=':9876',
                           lifetime='30m',
                           memory_limit = 0,
                           lifetime_stagger='4m',
                           lifetime_restart=True)
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

    cluster = SLURMCluster(queue = 'submit',
                           cores=1,
                           processes=1,
                           memory='8GB',
                           walltime='1:00:00',
                           log_directory=log_directory,
                           worker_extra_args=[
                           #   '--lifetime', '175m',
                           #   '--lifetime-stagger', '4m',
                           ],
                           scheduler_options={'dashboard_address':":9876"})
    client = Client(cluster)
    #client.register_worker_plugin(AddProcessPool())
    #print(cluster.job_script())
    cluster.adapt(
        minimum_jobs=1, maximum_jobs=500,
        worker_key=lambda state: state.address.split(':')[0],
        interval='10s'
    )
    
    #print("waiting 10 seconds for slurm jobs to spin up")
    #import time
    #time.sleep(10)

    #print(client.run(lambda dask_worker: str(dask_worker.executors)))
    print(cluster.job_script())


    return cluster, client
