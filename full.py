from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

import reader
import masks

import binMatch
import binEEC
import binJet

import plotMatch
import plotEEC
import plotJet

import matplotlib.pyplot as plt

fname = 'NANO_NANO.root'
x = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()

reco = reader.EECreader(x, 'RecoEEC')
gen = reader.EECreader(x, 'GenEEC')

trans = reader.transferreader(x, 'EECTransfer')

jets = reader.jetreader(x, 'selectedPatJetsAK4PFPuppi', 'SimonJets')
genjets = reader.genjetreader(x, 'ak4GenJetsNoNu', 'GenSimonJets', "EECTransfer")

muons = reader.muonreader(x, 'Muon')
HLT = x.HLT

evtMask = masks.evtMask(muons, HLT)

jetMask = evtMask & masks.jetMask(jets, muons)
genJetMask = evtMask & masks.genJetMask(genjets, muons)

# Htrans = binEEC.getHistPxP_bdiag()
# binEEC.fillHistTransferP(Htrans, reco, gen, trans, 1, jetMask)

# Hreco = binEEC.getHistP()
# Hcovreco = binEEC.getHistPxP()
# binEEC.fillHistP(Hreco, reco, 1, jetMask)
# binEEC.fillHistCovPxP(Hcovreco, reco, 1, jetMask)


# Hgen = binEEC.getHistP()
# Hcovgen = binEEC.getHistPxP()
# binEEC.fillHistP(Hgen, gen, 1, genJetMask)
# binEEC.fillHistCovPxP(Hcovgen, gen, 1, genJetMask)

# plotEEC.plotProjectedEEC(Hreco, Hcovreco, 2, "reco", show=False)
# plotEEC.plotBackground(Htrans, Hreco, Hcovreco, Hgen, Hcovgen, 2, "background", show=False)
# plotEEC.plotForward(Htrans, Hreco, Hgen, Hcovgen, 2, "transfered", show=False)
# plotEEC.plotProjectedEEC(Hgen, Hcovgen, 2, "gen", show=True, savefig="EEC2")
#plotEEC.plotProjectedEEC(Hgen, Hcovgen, 2, True, "EEC2.png", False, True)
'''
Hreco = binJet.getJetHist()
binJet.fillJetHist(Hreco, jets, 1, jetMask)

plotJet.plotJets(Hreco, 'pt')

Hreco = binMatch.getMatchHist()
binMatch.fillMatchHist(Hreco, jets, 1, jetMask)

Hgen = binMatch.getMatchHist()
binMatch.fillMatchHist(Hgen, genjets, 1, genJetMask)

plotMatch.plotMatchRate(Hgen, 'pt', -1, ylabel='Reconstruction Efficiency', savefig=None, show=False, match='match',   label='total match rate')
plotMatch.plotMatchRate(Hgen, 'pt', 1, ylabel='Reconstruction Efficiency', savefig=None, show=False, match='match',   label='exactly one match')
plotMatch.plotMatchRate(Hgen, 'pt', 2, ylabel='Reconstruction Efficiency', savefig=None, show=False, match='match',   label='exactly two matches')
plotMatch.plotMatchRate(Hgen, 'pt', 3, ylabel='Reconstruction Efficiency', savefig=None, show=False, match='match',   label='more than two matches')
plt.title("Gen particle reconstruction efficiency vs. particle $p_T$")
plt.legend()
plt.show()

plotMatch.plotMatchRate(Hreco, 'pt', -1, ylabel='Gen matching rate', savefig=None, show=False, match='match',   label='total match rate')
plotMatch.plotMatchRate(Hreco, 'pt', 1, ylabel='Gen matching rate', savefig=None, show=False, match='match',   label='exactly one match')
plotMatch.plotMatchRate(Hreco, 'pt', 2, ylabel='Gen matching rate', savefig=None, show=False, match='match',   label='exactly two matches')
plotMatch.plotMatchRate(Hreco, 'pt', 3, ylabel='Gen matching rate', savefig=None, show=False, match='match',   label='more than two matches')
plt.title("Reco particle gen-matching efficiency vs. particle $p_T$")
plt.legend()
plt.show()
'''
'''
plotMatch.plotMatchRate(Hreco, 'jetpt', -1, ylabel='Match Rate', savefig='matchRate_jetpt.png', show=True)
plotMatch.plotMatchRate(Hreco, 'eta', -1, ylabel='Match Rate', savefig='matchRate_eta.png', show=True)
plotMatch.plotMatchRate(Hreco, 'fracpt', -1, ylabel='Match Rate', savefig='matchRate_fracpt.png', show=True)

plotMatch.plotMatchRate(Hreco, 'pt', -1, ylabel='Match Rate', savefig='omatchRate_pt.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hreco, 'jetpt', -1, ylabel='Match Rate', savefig='omatchRate_jetpt.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hreco, 'eta', -1, ylabel='Match Rate', savefig='omatchRate_eta.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hreco, 'fracpt', -1, ylabel='Match Rate', savefig='omatchRate_fracpt.png', show=True, match='onmatch')


plotMatch.plotMatchRate(Hgen, 'pt', -1, ylabel='Reconstruction Efficiency', savefig='recoEff_pt.png', show=True)
plotMatch.plotMatchRate(Hgen, 'jetpt', -1, ylabel='Reconstruction Efficiency', savefig='recoEff_jetpt.png', show=True)
plotMatch.plotMatchRate(Hgen, 'eta', -1, ylabel='Reconstruction Efficiency', savefig='recoEff_eta.png', show=True)
plotMatch.plotMatchRate(Hgen, 'fracpt', -1, ylabel='Reconstruction Efficiency', savefig='recoEff_fracpt.png', show=True)

plotMatch.plotMatchRate(Hgen, 'pt', -1, ylabel='Reconstruction Efficiency', savefig='orecoEff_pt.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hgen, 'jetpt', -1, ylabel='Reconstruction Efficiency', savefig='orecoEff_jetpt.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hgen, 'eta', -1, ylabel='Reconstruction Efficiency', savefig='orecoEff_eta.png', show=True, match='onmatch')
plotMatch.plotMatchRate(Hgen, 'fracpt', -1, ylabel='Reconstruction Efficiency', savefig='orecoEff_fracpt.png', show=True, match='onmatch')
'''
