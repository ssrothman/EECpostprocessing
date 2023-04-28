from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import awkward as ak
import numpy as np

import read
import bin 

from importlib import reload

x = NanoEventsFactory.from_root("NANO_NANO.root", schemaclass=NanoAODSchema).events()

proj = read.getproj(x, "RecoEEC")
projdR = read.getprojdR(x, "RecoEEC")

HP = bin.getHistP()
bin.fillHistP(HP, projdR, proj, 1)
