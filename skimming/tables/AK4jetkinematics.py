import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

class AK4JetKinematicsTable:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "ak4jets"
    
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
        jets = objs.AK4Jets.jets
        thevals['Jpt'] = jets.pt[evtmask]
        thevals['Jeta'] = jets.eta[evtmask]
        thevals['Jphi'] = jets.phi[evtmask]
        
        #flavor info
        thevals['passLooseB'] = jets.passLooseB[evtmask]
        thevals['passMediumB'] = jets.passMediumB[evtmask]
        thevals['passTightB'] = jets.passTightB[evtmask]

        if hasattr(jets, 'hadronFlavour'):
            thevals['flav'] = jets.hadronFlavour[evtmask]
        
        #constituents
        thevals['nConstituents'] = jets.nConstituents[evtmask]

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