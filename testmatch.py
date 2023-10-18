from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

import reader
import masks
import weights

import binMatch
import plotMatch

import matplotlib.pyplot as plt

fname = '~/cmsdata/NANO_NANO.root'
x = NanoEventsFactory.from_root(fname, schemaclass=NanoAODSchema).events()

jets = reader.jetreader(x, 'selectedPatJetsAK4PFPuppi', 'SimonJets')
#genjets = reader.genjetreader(x, 'ak4GenJetsNoNu', 'GenSimonJets', "EECTransfer")

muons = reader.muonreader(x, 'Muon')
HLT = x.HLT

evtSel = masks.getEventSelection(muons, HLT)
jetSel = masks.getJetSelection(jets, muons, evtSel)
#genJetSel = masks.getGenJetSelection(genjets, muons, evtSel)

evtMask = evtSel.all(*evtSel.names)
jetMask = jetSel.all(*jetSel.names)
#genJetMask = genJetSel.all(*genJetSel.names)

evtWeight = weights.getEventWeight(x)
weight = evtWeight.weight()

hist_recomatch = binMatch.getMatchHist()
hist_genmatch = binMatch.getMatchHist()

#bin matching
binMatch.fillMatchHist(hist_recomatch, jets, weight, jetMask)
#binMatch.fillMatchHist(hist_genmatch, genJets, weight, genJetMask)
