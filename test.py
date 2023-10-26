from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import reading.reader

f = '../CMSSW_10_6_26/src/SRothman/NANO_NANO_NANO.root'
f = 'root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Oct19_2023_EECtest_take2/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/231019_192658/0000/NANO_NANO_NANO_99.root'
x = NanoEventsFactory.from_root(f).events()
print("GOT X")

from processing.EECProcessor import EECProcessor
p = EECProcessor(['EEC', 'NaiveEEC'], ['GenMatch', 'NaiveGenMatch'], 52, True, False)
q = p.process(x)

#print(q['EEC']['Htrans'].project("EECwtReco") - q['EEC']['HrecoPure'].project("EECwt")) 
print()
print("--"*20)
print()

rTransfer = reading.reader.transferreader(x, 'EECTransfer')
rGenEEC = reading.reader.EECreader(x, 'GenEEC')
rGenEECUNMATCH = reading.reader.EECreader(x, "GenEECUNMATCH")
rRecoEEC = reading.reader.EECreader(x, 'RecoEEC')
rRecoEECPU = reading.reader.EECreader(x, 'RecoEECPU')
rRecoJet = reading.reader.jetreader(x, '', 'SimonJets')
rGenJet = reading.reader.jetreader(x, '', 'GenSimonJets')

mask = np.abs(rRecoJet.simonjets.eta) < 1.0

import binning.binEEC_binwt
from importlib import reload

#print(q['EEC']['Htrans'].project("EECwtGen"))
#print(q['EEC']['HgenPure'].project("EECwt"))
def test(name):
    recoH = q[name]['HrecoPure']
    genH = q[name]['HgenPure']
    transrecoH = q[name]['Htrans'].project('ptReco', 'dRbinReco', 'EECwtReco')
    transgenH = q[name]['Htrans'].project('ptGen', 'dRbinGen', 'EECwtGen')

    recovals = recoH.values(flow=True)
    genvals = genH.values(flow=True)
    transrecovals = transrecoH.values(flow=True)
    transgenvals = transgenH.values(flow=True)

    print('--'*20)
    print(name)
    print("(reco==0) & (transreco!=0)", np.sum((recovals==0) & (transrecovals!=0)))
    print("(reco!=0) & (transreco==0)", np.sum((recovals!=0) & (transrecovals==0)))
    print("Max reco difference", np.max(np.abs(recovals - transrecovals)))
    print("target[1,0,30]", recovals[1,0,30])
    print("actual[1,0,30]", transrecovals[1,0,30])
    print("(gen==0) & (transgen!=0)", np.sum((genvals==0) & (transgenvals!=0)))
    print("(gen!=0) & (transgen==0)", np.sum((genvals!=0) & (transgenvals==0)))
    print()

#test('EEC')
#test('NaiveEEC')
