import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class MultiplicityBinner:
    def __init__(self, config, *ags, **kwargs):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

    def _getMultiplicityHist(self):
        return Hist(
            getAxis("pt", self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("partPtCategory", self._config['bins']),
            getAxis("DRaxis", self._config['bins']),
            getAxis("partSpecies", self._config['bins']),
            storage=Weight()
        )
    
    def _make_and_fill(self, rJet, mask, weight):
        hist = self._getMultiplicityHist()
        self._fill(hist, rJet, mask, weight)
        return hist

    def _fill(self, hist, rJet, mask, weight):
        parts = rJet.parts

        jetwt, _ = ak.broadcast_arrays(weight, rJet.jets.corrpt)

        pt, eta, weight, _ = ak.broadcast_arrays(rJet.jets.corrpt, 
                                                 rJet.jets.eta, 
                                                 weight, parts.pt)

        partPt = parts.pt
        partSpecies = ak.where(parts.charge !=0, 0, 
                               ak.where(parts.pdgid == 22, 1, 2))
        
        deta = np.abs(eta - parts.eta)
        dphi = np.abs(rJet.jets.phi - parts.phi)
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        dphi = ak.where(dphi < -np.pi, 2*np.pi + dphi, dphi)
        dR = np.sqrt(deta*eta + dphi*dphi)

        hist.fill(pt=squash(pt[mask]), 
                  eta=squash(np.abs(eta[mask])), 
                  partPtCategory=squash(partPt[mask]), 
                  DRaxis=squash(dR[mask]), 
                  partSpecies=squash(partSpecies[mask]),
                  weight=squash(weight[mask])
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        H = self._make_and_fill(readers.rRecoJet, jetMask, wt)

        return {'H' : H}
