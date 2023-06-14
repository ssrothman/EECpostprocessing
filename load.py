from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import EECProcessor

import plotEEC
import EECutil
from importlib import reload

def reloadAll():
    reload(plotEEC)
    reload(EECutil)

fname = 'NANO_NANO.root'

xALL = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()
Nevt = len(xALL)
print("Total number of events: ", Nevt)

x1 = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema, entry_start=0, entry_stop=Nevt//2).events()

x2 = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema, entry_start=Nevt//2, entry_stop=Nevt).events()

p = EECProcessor.EECProcessor()

out1 = p.process(x1)
out2 = p.process(x2)

outALL = p.process(xALL)
