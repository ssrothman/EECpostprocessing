import numpy as np
import awkward as ak
import warnings

from coffea.analysis_tools import PackedSelection

class PackedJetSelection:
    def __init__(self, evtSel=None):
        self.selection = PackedSelection()
        self.nums = None
        self.evtSel = evtSel

    def add(self, name, mask):
        if self.nums is None:
            self.nums = ak.num(mask)
            if self.evtSel is not None:
                for evtName in self.evtSel.names:
                    evtMask = self.evtSel.all(evtName)
                    newMask, _ = ak.broadcast_arrays(evtMask, mask)
                    self.selection.add(evtName, ak.flatten(newMask))
        else:
            assert ak.all(self.nums == ak.num(mask))
        self.selection.add(name, ak.flatten(mask))

    def all(self, *names):
        mask = self.selection.all(*names)
        return ak.unflatten(mask, self.nums)

    def any(self, *names):
        mask = self.selection.any(*names)
        return ak.unflatten(mask, self.nums)

    def require(self, **names):
        mask = self.selection.require(**names)
        return ak.unflatten(mask, self.nums)

    @property
    def names(self):
        return self.selection.names

def addMuonSelections(selection, rmu):
    selection.add("twomu", ak.count(rmu.muons.pt, axis=-1) >= 2)
    selection.add("mu1pt", rmu.muons[:,0].pt > 30)
    selection.add("mu2pt", rmu.muons[:,1].pt > 10)
    selection.add("mu1eta", np.abs(rmu.muons[:,0].eta) < 2.5)
    selection.add("mu2eta", np.abs(rmu.muons[:,1].eta) < 2.5)
    selection.add("oppsign", rmu.muons[:,0].charge * rmu.muons[:,1].charge < 0)
    return selection

def addTriggerSelections(selection, HLT):
    selection.add("IsoMu27", HLT.IsoMu27)
    #selection.add("DiMu", HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)
    return selection

def addZSelections(selection, rmu):
    Z = rmu.Zs
    selection.add("Zmass", (Z.mass > 70) & (Z.mass < 110))
    return selection

def getEventSelection(rmu, HLT):
    selection = PackedSelection()
    selection = addMuonSelections(selection, rmu)
    selection = addTriggerSelections(selection, HLT)
    selection = addZSelections(selection, rmu)
    return selection

def getJetSelection(rjet, rmu=None, evtSel = None):
    jets = rjet.jets
    selection = PackedJetSelection(evtSel)
    selection.add("pt", jets.pt > 30)
    selection.add("eta", np.abs(jets.eta) < 2.0)
    selection.add("jetId", (jets.jetId == 7) | (jets.pt > 50))
    if rmu is not None:
        muons = rmu.muons
        deta0 = jets.eta - ak.fill_none(muons[:,0].eta, 999)
        deta1 = jets.eta - ak.fill_none(muons[:,1].eta, 999)
        dphi0 = jets.phi - ak.fill_none(muons[:,0].phi, 999)
        dphi1 = jets.phi - ak.fill_none(muons[:,1].phi, 999)
        dphi0 = np.where(dphi0 > np.pi, dphi0 - 2*np.pi, dphi0)
        dphi0 = np.where(dphi0 < -np.pi, dphi0 + 2*np.pi, dphi0)
        dphi1 = np.where(dphi1 > np.pi, dphi1 - 2*np.pi, dphi1)
        dphi1 = np.where(dphi1 < -np.pi, dphi1 + 2*np.pi, dphi1)
        dR0 = np.sqrt(deta0*deta0 + dphi0*dphi0)
        dR1 = np.sqrt(deta1*deta1 + dphi1*dphi1)
        selection.add("muVeto", (dR0 > 0.4) & (dR1 > 0.4))
    return selection

def getGenJetSelection(rjet, rmu, evtSel = None):
    jets = rjet.jets
    muons = rmu.muons
    selection = PackedJetSelection(evtSel)
    selection.add("pt", jets.pt > 30)
    selection.add("eta", np.abs(jets.eta) < 2.4)
    selection.add("muVeto", (jets.delta_r(muons[:,0]) > 0.4)
                          & (jets.delta_r(muons[:,1]) > 0.4))
    return selection
