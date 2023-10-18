from coffea import processor

import reading.reader as reader

import selections.masks as masks
import selections.weights as weights

import binning.binMatch as binMatch

class MatchingProcessor(processor.ProcessorABC):
    def __init__(self, names):
        self.names = names

    def postprocess(self, accumulator):
        pass
    
    def process(self, events):
        #setup inputs
        jets = reader.jetreader(events, 'selectedPatJetsAK4PFPuppi', self.names[0]+"Particles")
        muons = reader.muonreader(events, "Muon")
        HLT = events.HLT

        #evtSel = masks.getEventSelection(muons, HLT)
        #jetSel = masks.getJetSelection(jets, muons, evtSel)

        #jetMask = jetSel.all(*jetSel.names)
        import numpy as np
        #jetMask = np.ones(len(HLT), dtype=bool)
        jetMask = np.abs(jets.simonjets.eta) < 2.0

        #evtWeight = weights.getEventWeight(events)
        #weight = evtWeight.weight()
        weight = np.ones(len(HLT))

        genjets = reader.jetreader(events, 'ak4GenJetsNoNu', self.names[0]+"GenParticles")
        #genJetMask = (genjets.jets.eta < 99999999) & evtSel.all(*evtSel.names)
        #genJetMask = jetMask
        genJetMask = np.abs(genjets.simonjets.eta) < 2.0

        result = {}
        for name in self.names:
            result[name] = binMatch.getMatchingHists(events, jetMask, genJetMask, weight, name)

        #return outputs
        return result
