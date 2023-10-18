import awkward as ak
import numpy as np
from coffea import processor

import reading.reader as reader

import selections.masks as masks
import selections.weights as weights

import binning.binMatch as binMatch
import binning.binEEC as binEEC
import binning.binJet as binJet

import matplotlib.pyplot as plt
import hist
import numpy as np
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor
import reading.reader as reader
import selections.masks as masks
import selections.weights as weights


import hist

class TransferProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def postprocess(self, accumulator):
        pass
    
    def process(self, events):
        #setup inputs
        jets = reader.jetreader(events, 'selectedPatJetsAK4PFPuppi', 'DefaultMatchParticles')
        muons = reader.muonreader(events, "Muon")
        HLT = events.HLT

        evtSel = masks.getEventSelection(muons, HLT)
        jetSel = masks.getJetSelection(jets, muons, evtSel)

        evtMask = evtSel.all(*evtSel.names)
        jetMask = jetSel.all(*jetSel.names)

        evtWeight = weights.getEventWeight(events)
        weight = evtWeight.weight()


        transferMask = jetMask[events.EECTransferBK.iReco]
        transfer = events.EECTransferproj.value2
        transfer = ak.unflatten(transfer, 484, axis=-1)[transferMask]
        transfer = ak.sum(ak.sum(transfer, axis=0), axis=0)

        gen = events.GenEECproj.value2
        gen = ak.unflatten(gen, 22, axis=-1)
        gen = gen[transferMask]
        gen = ak.sum(ak.sum(gen, axis=0), axis=0)

        #return outputs
        return {
            'transfer' : ak.to_numpy(transfer),
            'gen' : ak.to_numpy(gen)
        }
