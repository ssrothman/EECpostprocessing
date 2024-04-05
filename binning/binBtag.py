import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class BtagBinner:
    def __init__(self, config, config_tag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

        self._config['tag'] = vars(config_tag)

    def _getBtagHist(self):
        return Hist(
            getAxis("Beffpt", self._config['bins']),
            getAxis("Beffeta", self._config['bins']),
            getAxis("btag", self._config['bins'], "_tight"),
            getAxis("btag", self._config['bins'], "_medium"),
            getAxis("btag", self._config['bins'], "_loose"),
            getAxis("genflav", self._config['bins']),
            storage=Weight()
        )

    def _getBmatchHist(self):
        return Hist(
            getAxis("Beffpt", self._config['bins']),
            getAxis("Beffeta", self._config['bins']),
            getAxis("NumBMatch", self._config['bins']),
            storage=Weight()
        )
    
    def _make_and_fill_btag(self, rJet, mask, weight):
        hist = self._getBtagHist()
        self._fill_btag(hist, rJet, mask, weight)
        return hist

    def _fill_btag(self, hist, rJet, mask, weight):

        pt = rJet.jets.corrpt[mask]
        eta = np.abs(rJet.simonjets.jetEta[mask])

        btag_tight = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].tight
        btag_medium = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].medium
        btag_loose = rJet.CHSjets.btagDeepB[mask] > self._config['tag']['bwps'].loose

        btag_tight = ak.any(btag_tight, axis=-1)
        btag_medium = ak.any(btag_medium, axis=-1)
        btag_loose = ak.any(btag_loose, axis=-1)

        genflav = rJet.jets.hadronFlavour[mask]

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

    def _make_and_fill_bmatch(self, rJet, mask, weight):
        hist = self._getBmatchHist()
        self._fill_bmatch(hist, rJet, mask, weight)
        return hist

    def _fill_bmatch(self, hist, rJet, mask, weight):

        pt = rJet.jets.corrpt[mask]
        eta = np.abs(rJet.simonjets.jetEta[mask])

        numBMatch = ak.num(rJet.CHSjets.pt[rJet.CHSjets.pt>0][mask], axis=-1)

        weight, _ = ak.broadcast_arrays(weight, pt)

        hist.fill(
            pt = squash(pt),
            eta = squash(eta),
            NumBMatch = squash(numBMatch),
            weight = squash(weight)
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        Htag = self._make_and_fill_btag(readers.rRecoJet, jetMask, wt)
        Hmatch = self._make_and_fill_bmatch(readers.rRecoJet, jetMask, wt)
        return {
            'btag': Htag,
            'bmatch': Hmatch
        }
