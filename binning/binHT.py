import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class HTBinner:
    def __init__(self, config):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

    def _getHTHist(self):
        return Hist(
            getAxis("HT", self._config['bins']),
            storage=Weight()
        )
    
    def _make_and_fill(self, x, mask, weight):
        histGen = self._getHTHist()

        HTgen = x.LHE.HT
        histGen.fill(HTgen[mask], 
                     weight=weight[mask])

        return histGen

    def binAll(self, readers, jetMask, evtMask, wt):
        H = self._make_and_fill(readers.rRecoJet._x, evtMask, wt)

        return {'H' : H}
