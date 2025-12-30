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

    @property
    def name(self) -> str:
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