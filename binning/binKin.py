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

    def _getMuonHist(self, lead=False):
        return Hist(
            getAxis("MUpt", self._config['bins'], '_lead' if lead else '_sub'),
            getAxis("MUeta", self._config['bins'], '_lead' if lead else '_sub'),
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
            getAxis("btag", self._config['bins'], "_tight"),
            getAxis("btag", self._config['bins'], "_medium"),
            getAxis("btag", self._config['bins'], "_loose"),
            storage=Weight()
        )

    def getHTHist(self):
        return Hist(
            getAxis("HT", self._config['bins']),
            storage=Weight()
        )

    def getMETHist(self, nomask=False):
        return Hist(
            getAxis("METpt", self._config['bins'], '_nomask' if nomask else ''),
            getAxis("METsig", self._config['bins'], '_nomask' if nomask else ''),
            storage=Weight()
        )

    def getNBTagHist(self, domask=True):
        return Hist(
            getAxis("nBtag", self._config['bins'], '_nomask' if not domask else ''),
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

    def _make_and_fill_nbtag(self, rJet, evtMask, weight, domask):
        hist = self.getNBTagHist(domask)

        wp = vars(self._config['tag']['bwps'])[self._config['tag']['wp']]
        disc = rJet.CHSjets.btagDeepB
        passB = disc > wp
        nPass = ak.sum(ak.sum(passB, axis=-1), axis=-1)

        if domask:
            hist.fill(nPass[evtMask], weight=weight[evtMask])
        else:
            hist.fill(nPass, weight=weight)

        return hist

    def _make_and_fill_MET(self, rJet, evtMask, weight, domask):
        hist = self.getMETHist(not domask)

        METpt = rJet._x.MET.pt
        METsig = rJet._x.MET.significance

        if domask:
            hist.fill(METpt[evtMask], METsig[evtMask], weight=weight[evtMask])
        else:
            hist.fill(METpt, METsig, weight=weight)
        return hist

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
        hist_lead = self._getMuonHist(True)
        hist_sub = self._getMuonHist(False)

        self._fillMuon(hist_lead, rMu, evtMask, weight, lead=True)
        self._fillMuon(hist_sub, rMu, evtMask, weight, lead=False)

        return hist_lead, hist_sub

    def _fillMuon(self, hist, rMu, evtMask, weight, lead=True):

        mu0 = rMu.muons[:,0]
        mu1 = rMu.muons[:,1]
        leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
        submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

        weight = ak.broadcast_arrays(weight, leadmu.pt)[0]

        if lead:
            hist.fill(
                MUpt_lead=squash(leadmu.pt[evtMask]),
                MUeta_lead=squash(np.abs(leadmu.eta[evtMask])),
                weight=squash(weight[evtMask])
            )
        else: 
            hist.fill(
                MUpt_sub=squash(submu.pt[evtMask]),
                MUeta_sub=squash(np.abs(submu.eta[evtMask])),
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
            Zy=squash(np.abs(Z.y[evtMask])),
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

        wp_loose = self._config['tag']['bwps'].loose
        wp_medium = self._config['tag']['bwps'].medium
        wp_tight = self._config['tag']['bwps'].tight

        disc = rJet.CHSjets.btagDeepB[jetMask]

        btag_loose = disc > wp_loose
        btag_medium = disc > wp_medium
        btag_tight = disc > wp_tight

        btag_loose = ak.any(btag_loose, axis=-1)
        btag_medium = ak.any(btag_medium, axis=-1)
        btag_tight = ak.any(btag_tight, axis=-1)

        hist.fill(
            Jpt=squash(jet.corrpt),
            Jeta=squash(np.abs(jet.eta)),
            btag_loose=squash(btag_loose),
            btag_medium=squash(btag_medium),
            btag_tight=squash(btag_tight),
            weight=squash(weight)
        )

    def binAll(self, readers, jetMask, evtMask, wt):
        Hmu_lead, Hmu_sub = self._make_and_fill_mu(readers.rMu, evtMask, wt)
        HZ = self._make_and_fill_Z(readers.rMu, evtMask, wt)
        HNJet = self._make_and_fill_NJet(readers.rRecoJet, jetMask, evtMask, wt)
        HJet = self._make_and_fill_jet(readers.rRecoJet, jetMask, wt)
        HHT = self._make_and_fill_HT(readers.rRecoJet, evtMask, wt)
        Hrho = self._make_and_fill_rho(readers.rRecoJet, evtMask, wt)
        HPU = self._make_and_fill_PU(readers.rRecoJet, evtMask, wt)
        HMET = self._make_and_fill_MET(readers.rRecoJet, evtMask, wt, True)
        HMET_nomask = self._make_and_fill_MET(readers.rRecoJet, evtMask, wt, False)
        HNB = self._make_and_fill_nbtag(readers.rRecoJet, evtMask, wt, True)
        HNB_nomask = self._make_and_fill_nbtag(readers.rRecoJet, evtMask, wt, False)

        return {
            "Hmu_lead": Hmu_lead,
            "Hmu_sub": Hmu_sub,
            "HZ": HZ,
            "HNJet": HNJet,
            "HJet": HJet,
            'HHT': HHT,
            'Hrho' : Hrho,
            'HMET' : HMET,
            'HMET_nomask' : HMET_nomask,
            'HNB' : HNB,
            'HNB_nomask' : HNB_nomask,
            'HPU' : HPU,
        }

