import awkward as ak
import numpy as np
from coffea import processor

import reader

import masks
import weights

import binMatch
import binEEC
import binJet

class EECProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def postprocess(self, accumulator):
        pass
    
    def process(self, events):
        #setup inputs
        recoEEC = reader.EECreader(events, "RecoEEC")
        genEEC = reader.EECreader(events, 'GenEEC')
        transfer = reader.transferreader(events, 'EECTransfer')

        jets = reader.jetreader(events, 'selectedPatJetsAK4PFPuppi', 'SimonJets')
        genJets = reader.genjetreader(events, "ak4GenJetsNoNu", "GenSimonJets", 
                                   "EECTransfer")

        muons = reader.muonreader(events, "Muon")

        HLT = events.HLT

        #selections
        evtSel = masks.getEventSelection(muons, HLT)
        jetSel = masks.getJetSelection(jets, muons, evtSel)
        genJetSel = masks.getGenJetSelection(genJets, muons, evtSel)

        evtMask = evtSel.all(*evtSel.names)
        jetMask = jetSel.all(*jetSel.names)
        genJetMask = genJetSel.all(*genJetSel.names)

        #weights
        evtWeight = weights.getEventWeight(events)

        weight = evtWeight.weight()

        #setup empty EEC histograms
        hist_recoP = binEEC.getHistP()
        hist_recoPcov = binEEC.getHistPxP()
        hist_genP = binEEC.getHistP()
        hist_genPcov = binEEC.getHistPxP()

        hist_reco3 = binEEC.getHist3()
        hist_reco3cov = binEEC.getHist3x3()
        hist_gen3 = binEEC.getHist3()
        hist_gen3cov = binEEC.getHist3x3()

        hist_transferP = binEEC.getHistPxP_bdiag()
        hist_transfer3 = binEEC.getHist3x3()

        #bin EEC
        binEEC.fillHistP(hist_recoP, recoEEC, weight, jetMask)
        binEEC.fillHistP(hist_genP, genEEC, weight, genJetMask)
        binEEC.fillHistCovPxP(hist_recoPcov, recoEEC, weight, jetMask)
        binEEC.fillHistCovPxP(hist_genPcov, genEEC, weight, genJetMask)

        binEEC.fillHistRes3(hist_reco3, recoEEC, weight, jetMask)
        binEEC.fillHistRes3(hist_gen3, genEEC, weight, genJetMask)
        binEEC.fillHistCov3x3(hist_reco3cov, recoEEC, weight, jetMask)
        binEEC.fillHistCov3x3(hist_gen3cov, genEEC, weight, genJetMask)

        binEEC.fillHistTransferP(hist_transferP, recoEEC, genEEC, transfer, 
                                 weight, jetMask)
        binEEC.fillHistTransfer3(hist_transfer3, recoEEC, genEEC, transfer,
                                 weight, jetMask)

        #setup empty jet histogram
        hist_jet = binJet.getJetHist()

        #bin jet
        binJet.fillJetHist(hist_jet, jets, weight, jetMask)

        #setup empty matching histograms
        hist_recomatch = binMatch.getMatchHist()
        hist_genmatch = binMatch.getMatchHist()

        #bin matching
        binMatch.fillMatchHist(hist_recomatch, jets, weight, jetMask)
        binMatch.fillMatchHist(hist_genmatch, genJets, weight, genJetMask)

        #return outputs
        return {
            "recoP": hist_recoP,
            "recoPcov": hist_recoPcov,
            "genP": hist_genP,
            "genPcov": hist_genPcov,
            "reco3": hist_reco3,
            "reco3cov": hist_reco3cov,
            "gen3": hist_gen3,
            "gen3cov": hist_gen3cov,
            "transferP": hist_transferP,
            "transfer3": hist_transfer3,
            "jet": hist_jet,
            "recomatch": hist_recomatch,
            "genmatch": hist_genmatch,
        }
