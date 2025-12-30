import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_weight_variations, to_pa_table

class EventKinematicsTable:
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "events"
    
    def run_table(self, 
                  objs : AllObjects, 
                  evtsel : PackedSelection, 
                  jetsel : PackedJetSelection, 
                  weights : Weights):
        thevals = {}

        evtmask = evtsel.all()
        jetmask = jetsel.all()

        #MC properties
        if objs.isMC and hasattr(objs, 'LHE'):
            thevals['genHT'] = objs.LHE.HT[evtmask]
            thevals['genVpt'] = objs.LHE.Vpt[evtmask]

        #true PU
        if objs.isMC and hasattr(objs, 'PileupInfo'):
            thevals['nTrueInt'] = objs.PileupInfo.nTrueInt[evtmask]

        #global event properties
        thevals['rho'] = objs.rho[evtmask]
        thevals['MET'] = objs.MET.pt[evtmask]
        thevals['MET_phi'] = objs.MET.phi[evtmask]

        #btag multiplicities (for ttbar veto)
        thevals['numLooseB'] = ak.sum(objs.AK4Jets.jets.passLooseB[evtmask], axis=-1)
        thevals['numMediumB'] = ak.sum(objs.AK4Jets.jets.passMediumB[evtmask], axis=-1)
        thevals['numTightB'] = ak.sum(objs.AK4Jets.jets.passTightB[evtmask], axis=-1)

        #jet multiplicities
        thevals['numJets'] = ak.sum(jetmask, axis=-1)[evtmask]

        #Z kinematics
        thevals['Zpt'] = objs.Muons.Zs.pt[evtmask]
        thevals['Zmass'] = objs.Muons.Zs.mass[evtmask]
        thevals['Zphi'] = objs.Muons.Zs.phi[evtmask]
        thevals['Zy'] = objs.Muons.Zs.rapidity[evtmask]

        #muon kinematics
        mu0 = objs.Muons.muons[evtmask,0]
        mu1 = objs.Muons.muons[evtmask,1]
        leadmu : Any = ak.where(mu0.pt > mu1.pt, mu0, mu1)
        subleadmu : Any = ak.where(mu0.pt > mu1.pt, mu1, mu0)

        thevals['leadingMuPt'] = leadmu.pt
        thevals['leadingMuRawPt'] = leadmu.rawPt
        thevals['leadingMuEta'] = leadmu.eta
        thevals['leadingMuPhi'] = leadmu.phi
        thevals['leadingMuDxy'] = leadmu.dxy
        thevals['leadingMuDz'] = leadmu.dz
        thevals['leadingMuCharge'] = leadmu.charge

        thevals['subleadingMuPt'] = subleadmu.pt
        thevals['subleadingMuRawPt'] = subleadmu.rawPt
        thevals['subleadingMuEta'] = subleadmu.eta
        thevals['subleadingMuPhi'] = subleadmu.phi
        thevals['subleadingMuDxy'] = subleadmu.dxy
        thevals['subleadingMuDz'] = subleadmu.dz
        thevals['subleadingMuCharge'] = subleadmu.charge

        add_weight_variations(thevals, weights, evtmask)
        return to_pa_table(thevals)