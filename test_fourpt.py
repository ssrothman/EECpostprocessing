from coffea.nanoevents import NanoEventsFactory
import json
from RecursiveNamespace import RecursiveNamespace

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

evts = NanoEventsFactory.from_root('fourpt.root').events()

with open("configs/base.json", 'r') as f:
    config = RecursiveNamespace(**json.load(f))

with open("configs/ak8.json", 'r') as f:
    config.update(json.load(f))

with open("configs/inclusiveEEC.json", 'r') as f:
    config.update(json.load(f))

from reading.allreader import AllReaders

readers = AllReaders(evts, config,
                     noRoccoR = False,
                     noJER = True, noJEC = False)

ptmask = readers.rGenJet.jets.pt > 100
ptmask = ptmask[readers.rGenEEC.iJet]

res4 = ak.sum(ak.sum(readers.rGenEEC.res4[ptmask], axis=0), axis=0) #should have shape (nshape, RL, r, phi)

DATA = res4

res4 = ak.to_numpy(res4)

