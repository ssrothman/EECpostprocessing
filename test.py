from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

f = '../CMSSW_10_6_26/src/SRothman/NANO_NANO_NANO.root'
f = 'root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Oct19_2023_EECtest_take2/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/231019_192658/0000/NANO_NANO_NANO_99.root'
x = NanoEventsFactory.from_root(f).events()
print("GOT X")

from processing.EECProcessor import EECProcessor
p = EECProcessor(['EEC'], ['GenMatch'], 52)
q = p.process(x[:10000])
