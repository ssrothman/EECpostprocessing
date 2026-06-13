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
        thevals['pt'] = parts.pt[evtmask]
        thevals['eta'] = parts.eta[evtmask]
        thevals['phi'] = parts.phi[evtmask]
        thevals['pdgid'] = parts.pdgid[evtmask]
        thevals['charge'] = parts.charge[evtmask]
        thevals['dxy'] = parts.dxy[evtmask]
        thevals['dz'] = parts.dz[evtmask]
        thevals['puppiWeight'] = parts.puppiWeight[evtmask]
        thevals['fromPV'] = parts.fromPV[evtmask]

        #truth matching
        if hasattr(parts, 'matchPt'):
            thevals['matchPt'] = parts.matchPt[evtmask]
            thevals['matchEta'] = parts.matchEta[evtmask]
            thevals['matchPhi'] = parts.matchPhi[evtmask]
            thevals['matchCharge'] = parts.matchCharge[evtmask]
            thevals['matchTypes'] = parts.matchTypes[evtmask]
            thevals['nMatches'] = parts.nMatches[evtmask]
        
        #jet info
        jets = objs.RecoJets.jets
        thevals['Jpt'] = jets.pt[evtmask]
        thevals['Jeta'] = jets.eta[evtmask]
        thevals['Jphi'] = jets.phi[evtmask]

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
        
        #constituent kinematics
        parts = objs.GenJets.parts
        thevals['genPt'] = parts.pt[evtmask]
        thevals['genEta'] = parts.eta[evtmask]
        thevals['genPhi'] = parts.phi[evtmask]
        thevals['genPdgid'] = parts.pdgid[evtmask]
        thevals['genCharge'] = parts.charge[evtmask]

        #truth matching
        if hasattr(parts, 'nMatches'):
            thevals['nMatches'] = parts.nMatches[evtmask]
            thevals['matchTypes'] = parts.matchTypes[evtmask]
        
        #jet info
        jets = objs.GenJets.jets
        thevals['genJpt'] = jets.pt[evtmask]
        thevals['genJeta'] = jets.eta[evtmask]
        thevals['genJphi'] = jets.phi[evtmask]

        if hasattr(objs.GenJets.simonjets, 'genJetMatched'):
            thevals['genJetMatched'] = objs.GenJets.simonjets.genJetMatched[evtmask]

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