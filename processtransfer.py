import matplotlib.pyplot as plt
import hist
import numpy as np
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor
import reading.reader as reader
import selections.masks as masks
import selections.weights as weights
from processing.TransferProcessor import TransferProcessor
from processing.scaleout import setup_cluster_on_submit
from reading.files import get_rootfiles

hostid = "cmseos.fnal.gov"
rootpath = '/store/group/lpcpfnano/srothman/EEC_Jul17'
files = get_rootfiles(hostid, rootpath)
print(len(files))

if len(files) > 10:
    print("doing scale")
    cluster, client = setup_cluster_on_submit(1, 500)

    runner = Runner(
        executor=DaskExecutor(client=client, status=True),
        schema=NanoAODSchema
    )
else:
    runner = Runner(
        executor=IterativeExecutor(),
        schema=NanoAODSchema
    )

out = runner(
    {"DYJetsToLL" : files},
    treename='Events',
    processor_instance=TransferProcessor()
)


with open("trans.pkl", 'wb') as fout:
    import pickle
    pickle.dump(out, fout)

if len(files) > 10:
    client.close()
    cluster.close()
