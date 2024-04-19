import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class KinBinner:
    def __init__(self, config, config_tag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

        self._config['tag'] = vars(config_tag)

    def _getMuonHist(self):
        return Hist(
            getAxis("MUpt", self._config['bins']),
            getAxis("MUeta", self._config['bins']),
            storage=Weight()
        )

    def _getZHist(self):
        return Hist(
            getAxis("Zpt", self._config['bins']),
            getAxis("Zmass", self._config['bins']),
            getAxis("Zy", self._config['bins']),
            storage=Weight()
        )

    def _getNJetHist(self):
        return Hist(
            getAxis("NJet", self._config['bins']),
            storage=Weight()
        )

    def getJetHist(self):
        return Hist(
            getAxis("Jpt", self._config['bins']),
            getAxis("Jeta", self._config['bins']),
            storage=Weight()
        )

    def getHTHist(self):
        return Hist(
            getAxis("HT", self._config['bins']),
            storage=Weight()
        )

    def getRhoHist(self):
        return Hist(
            getAxis("rho", self._config['bins']),
            storage=Weight()
        )

    def getPUhist(self):
        return Hist(
            getAxis("nTruePU", self._config['bins']),
            storage=Weight()
        )

    def _make_and_fill_PU(self, rJet, evtMask, weight):
        hist = self.getPUhist()
        if hasattr(rJet._x, 'Pileup'):
            hist.fill(ak.values_astype(rJet._x.Pileup.nTrueInt[evtMask], 
                                       np.int32), 
                      weight=weight[evtMask])
        return hist

    def _make_and_fill_rho(self, rJet, evtMask, weight):
        hist = self.getRhoHist()
        self._fillRho(hist, rJet, evtMask, weight)
        return hist

    def _fillRho(self, hist, rJet, evtMask, weight):
        rho = rJet._x.fixedGridRhoFastjetAll[evtMask]
        hist.fill(rho, weight=weight[evtMask])

    def _make_and_fill_HT(self, rJet, evtMask, weight):
        hist = self.getHTHist()
        self._fillHT(hist, rJet, evtMask, weight)
        return hist

    def _fillHT(self, hist, rJet, evtMask, weight):
        if hasattr(rJet._x, 'LHE'):
            hist.fill(rJet._x.LHE.HT[evtMask], 
                      weight=weight[evtMask])

    def _make_and_fill_mu(self, rMu, evtMask, weight):
        hist = self._getMuonHist()
        self._fillMuon(hist, rMu, evtMask, weight)
        return hist

    def _fillMuon(self, hist, rMu, evtMask, weight):

        leadmu = rMu.muons[:,0]
        submu = rMu.muons[:,1]

        weight = ak.broadcast_arrays(weight, leadmu.pt)[0]

        hist.fill(
            MUpt=squash(leadmu.pt[evtMask]),
            MUeta=squash(leadmu.eta[evtMask]),
            weight=squash(weight[evtMask])
        )
         
        hist.fill(
            MUpt=squash(submu.pt[evtMask]),
            MUeta=squash(submu.eta[evtMask]),
            weight=squash(weight[evtMask])
        )
    def _make_and_fill_Z(self, rMu, evtMask, weight):
        hist = self._getZHist()
        self._fillZ(hist, rMu, evtMask, weight)
        return hist

    def _fillZ(self, hist, rMu, evtMask, weight):
        Z = rMu.Zs

        weight = ak.broadcast_arrays(weight, Z.pt)[0]

        hist.fill(
            Zpt=squash(Z.pt[evtMask]),
            Zmass=squash(Z.mass[evtMask]),
            Zy=squash(Z.y[evtMask]),
            weight=squash(weight[evtMask])
        )

    def _make_and_fill_NJet(self, rJet, jetMask, evtMask, weight):
        hist = self._getNJetHist()
        self._fillNJet(hist, rJet, jetMask, evtMask, weight)
        return hist

    def _fillNJet(self, hist, rJet, jetMask, evtMask, weight):

        nJet = ak.num(rJet.simonjets.jetPt[jetMask])[evtMask]
        weight = ak.broadcast_arrays(weight, evtMask)[0][evtMask]
        
        print(nJet)
        print('\tmin:', ak.min(nJet))
        print('\tmax:', ak.max(nJet))

        hist.fill(
            NJet=squash(nJet),
            weight=squash(weight)
        )

    def _make_and_fill_jet(self, rJet, jetMask, weight):
        hist = self.getJetHist()
        self._fillJet(hist, rJet, jetMask, weight)
        return hist

    def _fillJet(self, hist, rJet, jetMask, weight):
        jet = rJet.jets[jetMask]

        weight = ak.broadcast_arrays(weight, jet.corrpt)[0]

        hist.fill(
            Jpt=squash(jet.corrpt),
            Jeta=squash(jet.eta),
            weight=squash(weight)
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        Hmu = self._make_and_fill_mu(readers.rMu, evtMask, wt)
        HZ = self._make_and_fill_Z(readers.rMu, evtMask, wt)
        HNJet = self._make_and_fill_NJet(readers.rRecoJet, jetMask, evtMask, wt)
        HJet = self._make_and_fill_jet(readers.rRecoJet, jetMask, wt)
        HHT = self._make_and_fill_HT(readers.rRecoJet, evtMask, wt)
        Hrho = self._make_and_fill_rho(readers.rRecoJet, evtMask, wt)
        HPU = self._make_and_fill_PU(readers.rRecoJet, evtMask, wt)

        return {
            "Hmu": Hmu,
            "HZ": HZ,
            "HNJet": HNJet,
            "HJet": HJet,
            'HHT': HHT,
            'Hrho' : Hrho,
            'HPU' : HPU,
        }

