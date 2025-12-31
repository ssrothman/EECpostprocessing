import os
from correctionlib import CorrectionSet
from .PackedJetSelection import PackedJetSelection
from coffea.analysis_tools import PackedSelection
import awkward as ak
from typing import Any
from matplotlib.pyplot import gcf
import numpy as np

from skimming.objects.AllObjects import AllObjects

class StandardJetSelector:
    def __init__(self, cfg : dict):
        self._cfg = cfg

    def select_jets(self, allobjects : AllObjects, evtsel : PackedSelection, flags : dict) -> PackedJetSelection:
        selection = PackedJetSelection(evtsel)
        jets = allobjects.RecoJets.jets

        if self._cfg['minpt'] >= 0:
            selection.add(
                "jetpt",
                jets.pt >= self._cfg['minpt']
            )
        if self._cfg['maxeta'] >= 0:
            selection.add(
                "jeteta",
                np.abs(jets.eta) <= self._cfg['maxeta']
            )
        
        if self._cfg['jetID'] == 'loose':
            selection.add(
                "jetId",
                jets.jetIdLoose == 1
            )
        elif self._cfg['jetID'] == 'tight':
            selection.add(
                "jetId",
                jets.jetIdTight == 1
            )
        elif self._cfg['jetID'] == 'tightLepVeto':
            selection.add(
                "jetId",
                jets.jetIdLepVeto == 1
            )
        elif self._cfg['jetID'] == 'none':
            pass
        else:
            raise ValueError("Invalid jet ID: {}".format(self._cfg['jetID']))
        
        if self._cfg['puJetID'] == 'loose':
            selection.add(
                "puId",
                jets.puId >= 4
            )
        elif self._cfg['puJetID'] == 'medium':
            selection.add(
                "puId",
                jets.puId >= 6
            )
        elif self._cfg['puJetID'] == 'tight':
            selection.add(
                "puId",
                jets.puId >= 7
            )
        elif self._cfg['puJetID'] == 'none':
            pass
        else:
            raise ValueError("Invalid jet puID: {}".format(self._cfg['puJetID']))

        if self._cfg['muonOverlapDR'] >= 0:
            muons = allobjects.Muons.muons
            jets = allobjects.RecoJets.jets

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
            selection.add(
                "muVeto",
                (dR0 > self._cfg['muonOverlapDR'])
                & (dR1 > self._cfg['muonOverlapDR'])
            )

        if self._cfg['useVetoMap']:      
            vetocfg = self._cfg['jetvetomap']    
            vetmappath = vetocfg['path']
            if not vetmappath.startswith('/'):
                vetmappath = os.path.join(
                    os.path.dirname(__file__), '..', '..', vetmappath
                )  
            vetomap_cset = CorrectionSet.from_file(
                vetmappath
            )
            vetomap = vetomap_cset[vetocfg['name']]

            njet = ak.num(jets.eta)
            eta_flat = np.abs(ak.flatten(jets.eta))
            phi_flat = ak.flatten(jets.phi)
            phi_flat : Any = ak.where(phi_flat < -3.14159, -3.14159, phi_flat)
            phi_flat = ak.where(phi_flat > +3.14159, +3.14159, phi_flat)

            veto = vetomap.evaluate(
                vetocfg['whichmap'],
                eta_flat,
                phi_flat
            )

            passveto = (veto == 0)
            passveto = ak.unflatten(passveto, njet)

            selection.add("vetomap", passveto)

        return selection
