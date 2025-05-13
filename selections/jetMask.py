import awkward as ak
import numpy as np
from .PackedJetSelection import PackedJetSelection
from correctionlib import CorrectionSet

def getJetSelection(rjet, rmu, evtSel, config, isMC, verbose):
    if verbose:
        print("Setting up jet selections:")
    jets = rjet.jets
    selection = PackedJetSelection(evtSel)

    if config.jetSelection.pt > 0:
        if verbose:
            print("\tRequiring jet pt > %g"%config.jetSelection.pt)
        selection.add("pt", jets.corrpt > config.jetSelection.pt)
    if config.jetSelection.eta > 0:
        if verbose:
            print("\tRequiring jet |eta| < %g"%config.jetSelection.eta)
        selection.add("eta", np.abs(jets.eta) < config.jetSelection.eta)

    if config.jetSelection.jetID == 'loose':
        if verbose:
            print("\tRequiring jet loose ID")
        selection.add("jetId", jets.jetIdLoose > 0)
    elif config.jetSelection.jetID == 'tight':
        if verbose:
            print("\tRequiring jet tight ID")
        selection.add("jetId", jets.jetIdTight > 0)
    elif config.jetSelection.jetID == 'tightLepVeto':
        if verbose:
            print("\tRequiring jet tight ID + lep Veto")
        selection.add("jetId", jets.jetIdLepVeto > 0)
    elif config.jetSelection.jetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet ID: {}".format(config.jetSelection.jetID))

    if config.jetSelection.puJetID == 'loose':
        if verbose:
            print("\tRequiring jet loose PU ID")
        selection.add("puId", jets.puId >= 4)
    elif config.jetSelection.puJetID == 'medium':
        if verbose:
            print("\tRequiring jet medium PU ID")
        selection.add("puId", jets.puId >= 6)
    elif config.jetSelection.puJetID == 'tight':
        if verbose:
            print("\tRequiring jet tight PU ID")
        selection.add("puId", jets.puId >= 7)
    elif config.jetSelection.puJetID == 'none':
        pass
    else:
        raise ValueError("Invalid jet puID: {}".format(config.jetSelection.puJetID))

    if config.jetSelection.muonOverlapDR > 0:
        if verbose:
            print("\tRequiring jet pass muon overlap veto with threshold %g"%config.jetSelection.muonOverlapDR)

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
        if verbose:
            print("\tRequiring jet pass vetomap")
        vetomap_cset = CorrectionSet.from_file(config.jetvetomap.path)
        vetomap = vetomap_cset[config.jetvetomap.name]

        njet = ak.num(jets.eta)
        eta_flat = np.abs(ak.flatten(jets.eta))
        phi_flat = ak.flatten(jets.phi)
        phi_flat = ak.where(phi_flat < -3.14159, -3.14159, phi_flat)
        phi_flat = ak.where(phi_flat > +3.14159, +3.14159, phi_flat)

        veto = vetomap.evaluate(
            config.jetvetomap.whichmap,
            eta_flat,
            phi_flat
        )

        passveto = (veto == 0)

        passveto = ak.unflatten(passveto, njet)

        selection.add("vetomap", passveto)

    return selection
