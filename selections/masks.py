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

def addMuonSelections(selection, rmu, config):
    mu0 = rmu.muons[:,0]
    mu1 = rmu.muons[:,1]
    leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
    submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

    selection.add("twomu", ak.count(rmu.muons.pt, axis=-1) >= 2)
    selection.add("leadpt", leadmu.pt > config.leadpt)
    selection.add("subpt", submu.pt > config.subpt)
    selection.add("leadeta", np.abs(leadmu.eta) < config.leadeta)
    selection.add("subeta", np.abs(submu.eta) < config.subeta)
    if config.ID == "loose":
        selection.add("leadID", leadmu.looseId)
        selection.add("subID", submu.looseId)
    elif config.ID == "medium":
        selection.add("leadID", leadmu.mediumId)
        selection.add("subID", submu.mediumId)
    elif config.ID == "tight":
        selection.add("leadID", leadmu.tightId)
        selection.add("subID", submu.tightId)
    elif config.ID == 'none':
        pass
    else:
        raise ValueError("Invalid muon ID: {}".format(config.ID))
    if config.iso == 'loose':
        selection.add("leadiso", leadmu.pfIsoId >= 2)
        selection.add("subiso", submu.pfIsoId >= 2)
    elif config.iso == 'tight':
        selection.add("leadiso", leadmu.pfIsoId >= 4)
        selection.add("subiso", submu.pfIsoId >= 4)
    elif config.iso == 'none':
        pass
    else:
        raise ValueError("Invalid muon iso: {}".format(config.iso))

    if config.oppsign:
        selection.add("oppsign", (leadmu.charge * submu.charge) < 0)

    mask = selection.all(*selection.names)

    return selection

def addEventSelections(selection, HLT, config):
    selection.add("trigger", HLT[config.trigger])
    return selection

def addZSelections(selection, rmu, config):
    Z = rmu.Zs
    selection.add("Zmass", (Z.mass > config.mass[0]) & (Z.mass < config.mass[1]))
    selection.add("Zpt", Z.pt > config.minPt)
    selection.add("Zy", np.abs(Z.y) < config.maxY)
    return selection

def getEventSelection(rmu, HLT, config):
    selection = PackedSelection()
    selection = addMuonSelections(selection, rmu, config.muonSelection)
    selection = addZSelections(selection, rmu, config.Zselection)
    selection = addEventSelections(selection, HLT, config.eventSelection)
    return selection

def getJetSelection(rjet, rmu, evtSel, config):
    jets = rjet.jets
    selection = PackedJetSelection(evtSel)

    selection.add("pt", jets.pt > config.pt)
    selection.add("eta", np.abs(jets.eta) < config.eta)

    if config.jetID == 'loose':
        selection.add("jetId", jets.jetIdLoose > 0)
    elif config.jetID == 'tight':
        selection.add("jetId", jets.jetIdTight > 0)
    elif config.jetID == 'tightLepVeto':
        selection.add("jetId", jets.jetIdLepVeto > 0)
    elif config.jetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet ID: {}".format(config.jetID))

    if config.puJetID == 'loose':
        selection.add("puId", jets.puId >= 4)
    elif config.puJetID == 'medium':
        selection.add("puId", jets.puId >= 6)
    elif config.puJetID == 'tight':
        selection.add("puId", jets.puId >= 7)
    elif config.puJetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet puID: {}".format(config.puJetID))

    if config.muonOverlapDR > 0:
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
        selection.add("muVeto", (dR0>config.muonOverlapDR)&(dR1>config.muonOverlapDR))
    return selection

def getGenJetSelection(rjet, rmu, evtSel, config):
    jets = rjet.jets
    muons = rmu.muons
    selection = PackedJetSelection(evtSel)
    selection.add("pt", jets.pt > config.pt)
    selection.add("eta", np.abs(jets.eta) < config.eta)
    return selection
