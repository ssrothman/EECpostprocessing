from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import awkward as ak
import numpy as np

import readEEC
import binEEC
import reader

from importlib import reload

fname = '10k.root'
x = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()

reco = reader.reader(x, 'RecoEEC')
gen = reader.reader(x, 'GenEEC')
trans = reader.reader(x, 'EECTransf`er')
