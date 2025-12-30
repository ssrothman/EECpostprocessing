import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

class SimonJetKinematicsTable:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "jets"

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
        
        #jet kinematics
        jets = objs.RecoJets
        thevals['Jpt'] = jets.jets.pt[jetmask][evtmask]
        thevals['Jeta'] = jets.jets.eta[jetmask][evtmask]
        thevals['Jphi'] = jets.jets.phi[jetmask][evtmask]
        
        #flavor info
        thevals['passLooseB'] = jets.simonjets.passLooseB[jetmask][evtmask]
        thevals['passMediumB'] = jets.simonjets.passMediumB[jetmask][evtmask]
        thevals['passTightB'] = jets.simonjets.passTightB[jetmask][evtmask]

        if hasattr(jets.jets, 'hadronFlavour'):
            thevals['flav'] = jets.jets.hadronFlavour[jetmask][evtmask]
        
        #constituents
        thevals['nConstituents'] = jets.jets.nConstituents[jetmask][evtmask]
        thevals['nPassingConstituents'] = jets.simonjets.nPart[jetmask][evtmask]

        #CHS matching
        if hasattr(jets.simonjets, 'nCHS'):
            thevals['nCHS'] = jets.simonjets.nCHS[jetmask][evtmask]
            thevals['CHSpt'] = jets.simonjets.CHSpt[jetmask][evtmask]
            thevals['CHSeta'] = jets.simonjets.CHSeta[jetmask][evtmask]
            thevals['CHSphi'] = jets.simonjets.CHSphi[jetmask][evtmask]

        #genmatching
        if hasattr(jets.simonjets, 'jetMatchPt'):
            thevals['matchedPt'] = jets.simonjets.jetMatchPt[jetmask][evtmask]
            thevals['matchedEta'] = jets.simonjets.jetMatchEta[jetmask][evtmask]
            thevals['matchedPhi'] = jets.simonjets.jetMatchPhi[jetmask][evtmask]
            thevals['matched'] = jets.simonjets.jetMatched[jetmask][evtmask]

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals['Jpt']
        broadcast_all(thevals, shape_target)
        return to_pa_table(thevals)