import awkward as ak
import numpy as np
from .PackedJetSelection import PackedJetSelection

def getJetSelection(rjet, rmu, evtSel, config, isMC):
    jets = rjet.jets
    selection = PackedJetSelection(evtSel)

    selection.add("pt", jets.corrpt > config.pt)
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
