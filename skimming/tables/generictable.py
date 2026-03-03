import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects import EEC
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any, Literal, assert_never
import pyarrow as pa

from skimming.tables.common import add_common_vars, add_event_id, add_weight_variations, broadcast_all, to_pa_table

from typing import Sequence

class GenericTable:
    def __init__(self, object_name : str):
        self._object_name = object_name

    @property
    def name(self) -> str:
        return self._object_name
    
    def run_table(self, 
                objs : AllObjects, 
                evtsel : PackedSelection, 
                jetsel : PackedJetSelection, 
                weights : Weights):
        
        thevals = {}

        evtmask = evtsel.all()

        #keep some event-level properties
        #for post-hoc reweighting 
        #and playing with selections
        add_common_vars(thevals, objs, evtmask)

        theobj = getattr(objs, self._object_name)
        for field in theobj.fields:
            thevals[field] = getattr(theobj, field)[evtmask]

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        shape_target = thevals[theobj.fields[0]]
        broadcast_all(thevals, shape_target)

        return to_pa_table(thevals)
