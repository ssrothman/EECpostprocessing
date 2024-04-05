import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class BeffBinner:
    def __init__(self, config, config_tag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

        self._config['tag'] = vars(config_tag)

    def _getBeffHist(self):
        return Hist(
            getAxis("Beffpt", self._config['bins']),
            getAxis("Beffeta", self._config['bins']),
            getAxis("btag", self._config['bins'], "_tight"),
            getAxis("btag", self._config['bins'], "_medium"),
            getAxis("btag", self._config['bins'], "_loose"),
            getAxis("genflav", self._config['bins']),
            storage=Weight()
        )
    
    def _make_and_fill(self, rJet, mask, weight):
        hist = self._getBeffHist()
        self._fill(hist, rJet, mask, weight)
        return hist

    def _fill(self, hist, rJet, mask, weight):

        pt = rJet.CHSjets.pt[mask]
        eta = np.abs(rJet.CHSjets.eta[mask])
        btag_tight = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].tight
        btag_medium = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].medium
        btag_loose = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].loose
        genflav = rJet.CHSjets.hadronFlavour[mask]

        weight, _ = ak.broadcast_arrays(weight, pt)

        hist.fill(
            pt = squash(pt),
            eta = squash(eta),
            btag_tight = squash(btag_tight),
            btag_medium = squash(btag_medium),
            btag_loose = squash(btag_loose),
            genflav = squash(genflav),
            weight = squash(weight)
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        H = self._make_and_fill(readers.rRecoJet, jetMask, wt)
        return H
