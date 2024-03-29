import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class EventBinner:
    def __init__(self, config, config_btag, config_ctag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

        self._config['btag'] = vars(config_btag)
        self._config['ctag'] = vars(config_ctag)

    def _getJetHist(self):
        return Hist(
            getAxis("pt", self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("genflav", self._config['bins']),
            #getAxis("btag", self._config['bins']),
            #getAxis("ctag", self._config['bins']),
            hist.axis.Regular(20, 0, 1, name='B', label='B'),
            hist.axis.Regular(20, 0, 1, name='CvL', label='CvsL'),
            hist.axis.Regular(20, 0, 1, name='CvB', label='CvsB'),
            storage=Weight()
        )

    def _getMuonHist(self):
        return Hist(
            getAxis("MUpt", self._config['bins']),
            getAxis("MUeta", self._config['bins']),
            storage=Weight()
        )

    def _getZHist(self):
        return Hist(
            getAxis("Zmass", self._config['bins']),
            getAxis("Zpt", self._config['bins']),
            getAxis("Zy", self._config['bins']),
            storage=Weight()
        )

    def _make_and_fill_jet(self, rJet, mask, weight):
        jetHist = self._getJetHist()
        self._fillJet(jetHist, rJet, mask, weight)
        return jetHist

    def _fillJet(self, jetHist, rJet, mask, weight):
        weight, _ = np.broadcast_arrays(weight, rJet.CHSjets.pt)

        jetHist.fill(
            pt = squash(rJet.CHSjets.pt[mask]),
            eta = squash(np.abs(rJet.CHSjets.eta[mask])),
            genflav = squash(rJet.CHSjets.hadronFlavour[mask]),
            B = squash(rJet.CHSjets.btagDeepB[mask]),
            CvL = squash(rJet.CHSjets.btagDeepCvL[mask]),
            CvB = squash(rJet.CHSjets.btagDeepCvB[mask]),
            #btag = squash(passBtag(rJet, slice(None), mask, self._config['btag'])),
            #ctag = squash(passCtag(rJet, slice(None), mask, self._config['ctag'])),
            #B = squash(getBdisc(rJet, slice(None), mask, self._config['btag'])),
            #CvL = squash(getCvLdisc(rJet, slice(None), mask, self._config['ctag'])),
            #CvB = squash(getCvBdisc(rJet, slice(None), mask, self._config['ctag'])),
            weight = squash(weight[mask])
        )

    def _make_and_fill_muon(self, rMu, mask, weight):
        leadMuHist = self._getMuonHist()
        subMuHist = self._getMuonHist()
        self._fillMuon(leadMuHist, subMuHist, rMu, mask, weight)
        return leadMuHist, subMuHist

    def _fillMuon(self, leadMuHist, subMuHist, rMu, mask, weight):
        mu0 = rMu.muons[:,0]
        mu1 = rMu.muons[:,1]
        leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
        submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

        leadMuHist.fill(
            MUpt = squash(leadmu.pt[mask]),
            MUeta = squash(np.abs(leadmu.eta[mask])),
            weight = squash(weight[mask])
        )
        subMuHist.fill(
            MUpt = squash(submu.pt[mask]),
            MUeta = squash(np.abs(submu.eta[mask])),
            weight = squash(weight[mask])
        )

    def _make_and_fill_z(self, rMu, mask, weight):
        zHist = self._getZHist()
        self._fillZ(zHist, rMu, mask, weight)
        return zHist

    def _fillZ(self, zHist, rMu, mask, weight):
        weight, _ = np.broadcast_arrays(weight, rMu.Zs.pt)
        zHist.fill(
            Zmass = squash(rMu.Zs.mass[mask]),
            Zpt = squash(rMu.Zs.pt[mask]),
            Zy = squash(np.abs(rMu.Zs.y[mask])),
            weight = squash(weight[mask])
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        HrecoJet = self._make_and_fill_jet(readers.rRecoJet, jetMask, wt)
        HleadMu, HsubMu = self._make_and_fill_muon(readers.rMu, evtMask, wt)
        HZ = self._make_and_fill_z(readers.rMu, evtMask, wt)

        return {
            'recoJet': HrecoJet,
            'leadmu': HleadMu,
            'submu': HsubMu,
            'Z': HZ,
            'sumwt' : ak.sum(wt[evtMask]),
        }
