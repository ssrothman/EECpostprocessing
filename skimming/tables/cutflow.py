import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any

class CutflowTable:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "cutflow"

    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        

        cutflow = {
            "evtflow" : {
                "names" : [],
                "counts" : []
            },
            "jetflow" : {
                "names" : [],
                "counts" : []
            }
        }


        # Event-level cutflow
        nomwt : Any = weights.weight()

        cuts_so_far = []

        # Special field for raw event count
        cutflow['evtflow']['names'].append('raw')
        cutflow['evtflow']['counts'].append(float(len(objs.event)))

        for name in evtsel.names:
            cuts_so_far.append(name)
            mask = evtsel.all(*cuts_so_far)
            wts = nomwt[mask]
            count = np.sum(wts)
            cutflow["evtflow"]["names"].append(name)
            cutflow["evtflow"]["counts"].append(float(count))
        
        # Jet-level cutflow
        jetnomwt : Any = ak.broadcast_arrays(nomwt, objs.RecoJets.jets.pt)[0]

        cuts_so_far = []
        for name in jetsel.names:
            cuts_so_far.append(name)
            mask = jetsel.all(*cuts_so_far)
            wts = jetnomwt[mask]
            count = ak.sum(wts, axis=None)
            cutflow["jetflow"]["names"].append(name)
            cutflow["jetflow"]["counts"].append(float(count))

        return cutflow