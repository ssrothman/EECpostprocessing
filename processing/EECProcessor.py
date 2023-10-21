import awkward as ak
import numpy as np
from coffea import processor

import reading.reader as reader

import selections.masks as masks
import selections.weights as weights

import binning.binEEC as binEEC
import binning.binEEC_binwt as binEEC_binwt

class EECProcessor(processor.ProcessorABC):
    def __init__(self, names, matchNames, nDR, binwt):
        self.names = names
        self.matchNames = matchNames
        self.nDR = nDR
        self.binner = binEEC_binwt if binwt else binEEC

    def postprocess(self, accumulator):
        pass
    
    def process(self, events):
        print("top of process")
        #setup inputs
        jets = reader.jetreader(events, 'selectedPatJetsAK4PFPuppi', "SimonJets")
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

        #return outputs
        result = {}
        print("just before loop")

        for name, matchName in zip(self.names, self.matchNames):
            print("doing", name, matchName)
            result[name] = self.binner.doAll(
                    events, '%sTransfer'%name, 'Reco%s'%name, 'Gen%s'%name,
                    '%sParticles'%matchName, '%sGenParticles'%matchName,
                    self.nDR, weight, jetMask)

        return result
