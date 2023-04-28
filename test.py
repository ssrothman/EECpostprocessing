from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import awkward as ak
import numpy as np

import read
import bin 
import reader

from importlib import reload

x = NanoEventsFactory.from_root("NANO_NANO.root", schemaclass=NanoAODSchema).events()

reco = reader.reader(x, 'RecoEEC')
gen = reader.reader(x, 'GenEEC')
trans = reader.reader(x, 'EECTransfer')

HTP = bin.getHistPxP_bdiag()
