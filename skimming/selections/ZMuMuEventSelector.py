from coffea.analysis_tools import PackedSelection
import awkward as ak
from typing import Any
from matplotlib.pyplot import gcf
import numpy as np

from skimming.objects.AllObjects import AllObjects

class ZMuMuEventSelector:
    def __init__(self, cfg : dict):
        self._cfg = cfg

    def select_events(self, allobjects : AllObjects, flags : dict) -> PackedSelection:
        selection = PackedSelection()
       
        #hack to build orthogonal HT < 70 sample
        if 'genHT' in flags:
            selection.add(
                "genHT",
                allobjects.LHE.HT <= flags['genHT']
            )

        selection = self._addMuonSelections(selection, allobjects)
        selection = self._addZSelections(selection, allobjects)
        selection = self._addGlobalSelections(selection, allobjects)

        return selection

    def _addMuonSelections(self, 
                           selection : PackedSelection,
                           objects : AllObjects):
        mucfg = self._cfg['muons'] 

        muons = objects.Muons.muons

        mu0 = muons[:,0]
        mu1 = muons[:,1]

        leadmu : Any = ak.where( #Any type to appease pyright
            mu0.pt > mu1.pt,
            mu0,
            mu1
        )
        submu : Any = ak.where( #Any type to appease pyright
            mu0.pt > mu1.pt,   
            mu1,
            mu0
        )

        selection.add(
            'twomu', 
            ~ak.is_none(mu0) & ~ak.is_none(mu1)
        )

        if mucfg['leadpt'] >= 0:
            selection.add(
                "leadpt", 
                leadmu.pt >= mucfg['leadpt']
            )
        if mucfg['subpt'] >= 0:
            selection.add(
                "subpt", 
                submu.pt >= mucfg['subpt']
            )
        if mucfg['leadeta'] >= 0:
            selection.add(
                "leadeta", 
                np.abs(leadmu.eta) < mucfg['leadeta']
            )
        if mucfg['subeta'] >= 0:
            selection.add(
                "subeta", 
                np.abs(submu.eta) < mucfg['subeta']
            )
        if mucfg['ID'] == "loose":
            selection.add(
                "muonlooseID",
                leadmu.looseId & submu.looseId
            )
        elif mucfg['ID'] == "medium":
            selection.add(
                "muonmediumID",
                leadmu.mediumId & submu.mediumId
            )
        elif mucfg['ID'] == "tight":
            selection.add(
                "muontightID",
                leadmu.tightId & submu.tightId
            )
        elif mucfg['ID'] == 'none':
            pass
        else:
            raise ValueError("Invalid muon ID: {}".format(mucfg['ID']))
        
        if mucfg['iso'] == 'loose':
            selection.add(
                "muonlooseiso",
                (leadmu.pfIsoId >= 2) & (submu.pfIsoId >= 2)
            )
        elif mucfg['iso'] == 'tight':
            selection.add(
                "muontightiso",
                (leadmu.pfIsoId >= 4) & (submu.pfIsoId >= 4)
            )
        elif mucfg['iso'] == 'none':
            pass
        else:
            raise ValueError("Invalid muon iso: {}".format(mucfg['iso']))

        if mucfg['dxy'] >= 0:
            selection.add(
                "muondxy",
                (np.abs(leadmu.dxy) < mucfg['dxy'])
                & (np.abs(submu.dxy) < mucfg['dxy'])
            )
        if mucfg['dz'] >= 0:
            selection.add(
                "muondz",
                (np.abs(leadmu.dz) < mucfg['dz'])
                & (np.abs(submu.dz) < mucfg['dz'])
            )
        
        if mucfg['oppsign']:
            selection.add(
                "oppsign",
                leadmu.charge != submu.charge
            )

        return selection
    
    def _addZSelections(self, 
                        selection : PackedSelection,
                        objects : AllObjects):
        Zcfg = self._cfg['Zs']

        Zs = objects.Muons.Zs

        if Zcfg['massWindow'] >= 0:
            upper = Zcfg['polemass'] + Zcfg['massWindow']
            lower = Zcfg['polemass'] - Zcfg['massWindow']
            selection.add(
                "Zmasswindow",
                (Zs.mass >= lower) & (Zs.mass <= upper)
            )

        if Zcfg['minpt'] >= 0:
            selection.add(
                "Zminpt",
                Zs.pt >= Zcfg['minpt']
            )
        if Zcfg['maxY'] >= 0:
            selection.add(
                "ZmaxY",
                np.abs(Zs.rapidity) <= Zcfg['maxY']
            )
        return selection
    
    def _addGlobalSelections(self,
                             selection : PackedSelection,
                             objects : AllObjects):
        
        gcfg = self._cfg['global']

        if gcfg['trigger']:
            selection.add(
                "trigger",
                objects.HLT[gcfg['trigger']]
            )
        if gcfg['maxMETpt'] >= 0:
            selection.add(
                "maxMETpt",
                objects.MET.pt <= gcfg['maxMETpt']
            )
        if gcfg['maxNumBtag'] >= 0:
            level = gcfg['maxNumBtag_level']
            passB = objects.AK4Jets.jets['pass%sB' % (level)]
            nBtag = ak.sum(passB, axis=1)
            selection.add(
                "maxNumBtag",
                nBtag <= gcfg['maxNumBtag']
            )
        if gcfg['noiseFilters']:
            filtermask = np.ones(len(objects.event), dtype=bool)
            for flag in gcfg['noiseFilters']:
                filtermask = filtermask & objects.Flag[flag]
            selection.add(
                "noiseFilters",
                filtermask
            )
        return selection