import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

class ConstituentKinematicsTable:
    def __init__(self):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return "parts"
    
    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)
        
        #constituent kinematics
        parts = objs.RecoJets.parts
        thevals['pt'] = parts.pt[jetmask][evtmask]
        thevals['eta'] = parts.eta[jetmask][evtmask]
        thevals['phi'] = parts.phi[jetmask][evtmask]
        thevals['pdgid'] = parts.pdgid[jetmask][evtmask]
        thevals['charge'] = parts.charge[jetmask][evtmask]
        thevals['dxy'] = parts.dxy[jetmask][evtmask]
        thevals['dz'] = parts.dz[jetmask][evtmask]
        thevals['puppiWeight'] = parts.puppiWeight[jetmask][evtmask]
        thevals['fromPV'] = parts.fromPV[jetmask][evtmask]

        #truth matching
        if hasattr(parts, 'matchPt'):
            thevals['matchPt'] = parts.matchPt[jetmask][evtmask]
            thevals['matchEta'] = parts.matchEta[jetmask][evtmask]
            thevals['matchPhi'] = parts.matchPhi[jetmask][evtmask]
            thevals['matchCharge'] = parts.matchCharge[jetmask][evtmask]
            thevals['matchTypes'] = parts.matchTypes[jetmask][evtmask]
            thevals['nMatches'] = parts.nMatches[jetmask][evtmask]
        
        #jet info
        jets = objs.RecoJets.jets
        thevals['Jpt'] = jets.pt[jetmask][evtmask]
        thevals['Jeta'] = jets.eta[jetmask][evtmask]
        thevals['Jphi'] = jets.phi[jetmask][evtmask]

        if hasattr(objs.RecoJets.simonjets, 'jetMatched'):
            thevals['jetMatched'] = objs.RecoJets.simonjets.jetMatched[jetmask][evtmask]

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals['pt']
        broadcast_all(thevals, shape_target)

        return to_pa_table(thevals)


class GenConstituentKinematicsTable:
    def __init__(self):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return "genparts"
    
    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)
        
        noReco = objs.GenJets.simonjets.iReco < 0
        clippedReco = ak.where(noReco, 0, objs.GenJets.simonjets.iReco)
        genjetmask = jetmask[clippedReco]
        genjetmask = ak.where(noReco, False, genjetmask)

        #constituent kinematics
        parts = objs.GenJets.parts
        thevals['genPt'] = parts.pt[genjetmask][evtmask]
        thevals['genEta'] = parts.eta[genjetmask][evtmask]
        thevals['genPhi'] = parts.phi[genjetmask][evtmask]
        thevals['genPdgid'] = parts.pdgid[genjetmask][evtmask]
        thevals['genCharge'] = parts.charge[genjetmask][evtmask]

        #truth matching
        if hasattr(parts, 'nMatches'):
            thevals['nMatches'] = parts.nMatches[genjetmask][evtmask]
            thevals['matchTypes'] = parts.matchTypes[genjetmask][evtmask]

        #jet info
        jets = objs.GenJets.jets
        thevals['genJpt'] = jets.pt[genjetmask][evtmask]
        thevals['genJeta'] = jets.eta[genjetmask][evtmask]
        thevals['genJphi'] = jets.phi[genjetmask][evtmask]

        if hasattr(objs.GenJets.simonjets, 'genJetMatched'):
            thevals['genJetMatched'] = objs.GenJets.simonjets.genJetMatched[genjetmask][evtmask]
            thevals['iReco'] = objs.GenJets.simonjets.iReco[genjetmask][evtmask]

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals['genPt']
        broadcast_all(thevals, shape_target)

        return to_pa_table(thevals)