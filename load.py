from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

import EECProcessor

fname = 'NANO_NANO.root'
x = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema, entry_stop=1000).events()

p = EECProcessor.EECProcessor()
out = p.process(x)
