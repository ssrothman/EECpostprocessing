from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

f = '../CMSSW_10_6_26/src/SRothman/NANO_NANO_NANO.root'
f = 'root://cmseos.fnal.gov///store/group/lpcpfnano/srothman/Oct19_2023_EECtest/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/231019_155522/0000/NANO_NANO_NANO_7.root'
x = NanoEventsFactory.from_root(f).events()
print("GOT X")

from processing.EECProcessor import EECProcessor
p = EECProcessor(['EEC', 'NaiveEEC'], ['GenMatch', 'NaiveGenMatch'], 52)
q = p.process(x[:100])

y = x[:100]

import reading.reader

rTransfer = reading.reader.transferreader(y, 'EECTransfer')
rGenEEC = reading.reader.EECreader(y, 'GenEEC')
rRecoEEC = reading.reader.EECreader(y, 'RecoEEC')
rRecoEECPU = reading.reader.EECreader(y, 'RecoEECPU')

transferval = rTransfer.proj
mask = ak.num(transferval) > 0
transferval = transferval[mask]

iReco = ak.local_index(transferval, axis=3)
iGen = ak.local_index(transferval, axis=2)

reco = (rRecoEEC.proj - rRecoEECPU.proj)[mask]
gen = rGenEEC.proj[mask]

transferRECO = transferval
transferGEN = gen[iGen]
