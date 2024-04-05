import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class MultiplicityBinner:
    def __init__(self, config):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

    def _getMultiplicityHist(self):
        return Hist(
            getAxis("pt", self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("partPt", self._config['bins']),
            getAxis("DRaxis", self._config['bins']),
            getAxis("partCharge", self._config['bins']),
            storage=Weight()
        )
    
    def _make_and_fill(self, rJet, mask, weight):
        hist = self._getMultiplicityHist()
        sumwt = self._fill(hist, rJet, mask, weight)
        return hist, sumwt

    def _fill(self, hist, rJet, mask, weight):
        parts = rJet.parts

        jetwt, _ = ak.broadcast_arrays(weight, rJet.jets.pt)

        pt, eta, weight, _ = ak.broadcast_arrays(rJet.jets.pt, rJet.jets.eta, 
                                                 weight, parts.pt)

        partPt = parts.pt
        partCharge = parts.charge!=0
        
        deta = np.abs(eta - parts.eta)
        dphi = np.abs(rJet.jets.phi - parts.phi)
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        dphi = ak.where(dphi < -np.pi, 2*np.pi + dphi, dphi)
        dR = np.sqrt(deta*eta + dphi*dphi)

        hist.fill(pt=squash(pt[mask]), 
                  eta=squash(eta[mask]), 
                  partPt=squash(partPt[mask]), 
                  DRaxis=squash(dR[mask]), 
                  partCharge=squash(partCharge[mask]),
                  weight=squash(weight[mask])
        )
        return ak.sum(jetwt[mask], axis=None)

    def binAll(self, readers, jetMask, evtMask, wt):
        H, sumwt = self._make_and_fill(readers.rRecoJet, jetMask, wt)

        return {'H' : H, 'sumwt' : sumwt}
