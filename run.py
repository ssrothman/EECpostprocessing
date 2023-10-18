from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from processing.EECProcessor import EECProcessor
from processing.scaleout import setup_cluster_on_submit
from reading.files import get_rootfiles


hostid = "cmseos.fnal.gov"
rootpath = '/store/group/lpcpfnano/srothman/EEC_Jul09_take2/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/'
cluster, client = setup_cluster_on_submit(1, 250)

files = get_rootfiles(hostid, rootpath)
runner = Runner(
    executor=DaskExecutor(client=client, status=True),
    schema=NanoAODSchema
)

out = runner(
    {"DYJetsToLL" : files}, 
    treename='Events',
    processor_instance=EECProcessor()
)

with open("test.pkl", 'wb') as fout:
    import pickle
    pickle.dump(out, fout)

cluster.close()
client.close()
