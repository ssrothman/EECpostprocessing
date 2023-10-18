import os

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def setup_cluster_on_submit(minjobs, maxjobs, path=None):
    import random

    if path is None:
        uuid = random.getrandbits(64)
        log_directory = 'logs/%016x'%uuid
    else:
        log_directory = os.path.join(path,'logs')

    print("saving logs at",log_directory)

    cluster = SLURMCluster(queue = 'submit,submit-gpu,submit-gpu1080',
                           cores=1,
                           processes=1,
                           memory='32GB',
                           walltime='01:00:00',
                           log_directory=log_directory)
    cluster.adapt(minimum_jobs = minjobs, maximum_jobs = maxjobs)
    #cluster.scale(maxjobs)
    client = Client(cluster)
    return cluster, client
