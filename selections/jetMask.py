import awkward as ak
import numpy as np
from .PackedJetSelection import PackedJetSelection
from correctionlib import CorrectionSet

def getJetSelection(rjet, rmu, evtSel, config, isMC):
    jets = rjet.jets
    selection = PackedJetSelection(evtSel)

    if config.jetSelection.pt > 0:
        selection.add("pt", jets.pt > config.jetSelection.pt)
    if config.jetSelection.eta > 0:
        selection.add("eta", np.abs(jets.eta) < config.jetSelection.eta)

    if config.jetSelection.jetID == 'loose':
        selection.add("jetId", jets.jetIdLoose > 0)
    elif config.jetSelection.jetID == 'tight':
        selection.add("jetId", jets.jetIdTight > 0)
    elif config.jetSelection.jetID == 'tightLepVeto':
        selection.add("jetId", jets.jetIdLepVeto > 0)
    elif config.jetSelection.jetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet ID: {}".format(config.jetSelection.jetID))

    if config.jetSelection.puJetID == 'loose':
        selection.add("puId", jets.puId >= 4)
    elif config.jetSelection.puJetID == 'medium':
        selection.add("puId", jets.puId >= 6)
    elif config.jetSelection.puJetID == 'tight':
        selection.add("puId", jets.puId >= 7)
    elif config.jetSelection.puJetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet puID: {}".format(config.jetSelection.puJetID))

    if config.jetSelection.muonOverlapDR > 0:
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
        selection.add("muVeto", (dR0>config.jetSelection.muonOverlapDR)&(dR1>config.jetSelection.muonOverlapDR))

    if config.jetSelection.useVetoMap:
        vetomap_cset = CorrectionSet.from_file(config.jetvetomap.path)
        vetomap = vetomap_cset[config.jetvetomap.name]

        njet = ak.num(jets.eta)
        eta_flat = np.abs(ak.flatten(jets.eta))
        phi_flat = ak.flatten(jets.phi)

        veto = vetomap.evaluate(
            config.jetvetomap.whichmap,
            eta_flat,
            phi_flat
        )

        passveto = (veto == 0)

        passveto = ak.unflatten(passveto, njet)

        selection.add("vetomap", passveto)

    return selection
