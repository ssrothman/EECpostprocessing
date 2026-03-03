import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import PackedSelection
from skimming.selections.PackedJetSelection import PackedJetSelection
from typing import Any
import pyarrow as pa

from skimming.tables.common import add_event_id, add_weight_variations, to_pa_table

class EventKinematicsTable:
    def __init__(self):
        pass

    @classmethod
    @property
    def name(cls) -> str:
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

        # third muon
        thevals['nMu'] = ak.num(objs.Muons.muons[evtmask], axis=1)
        mu3s = objs.Muons.muons[evtmask, 2:]
        mu3s = ak.pad_none(mu3s, 1, axis=1) # ensure at least one entry to avoid empty array issues
        thevals['thirdMuPt'] = ak.fill_none(mu3s.pt[:,0], 0.0)
        thevals['thirdMuEta'] = ak.fill_none(mu3s.eta[:,0], -999.0)
        thevals['thirdMuPhi'] = ak.fill_none(mu3s.phi[:,0], -999.0)
        thevals['thirdMuDxy'] = ak.fill_none(mu3s.dxy[:,0], -999.0)
        thevals['thirdMuDz'] = ak.fill_none(mu3s.dz[:,0], -999.0)
        thevals['thirdMuCharge'] = ak.fill_none(mu3s.charge[:,0], -999)  # Keep as -999 for charge

        # electrons
        thevals['nEle'] = ak.num(objs.Electrons.electrons[evtmask], axis=1)
        eles = objs.Electrons.electrons[evtmask]
        eles = ak.pad_none(eles, 1, axis=1) # ensure at least one entry to avoid empty array issues
        thevals['leadingElePt'] = ak.fill_none(eles.pt[:,0], 0.0)
        thevals['leadingEleEta'] = ak.fill_none(eles.eta[:,0], -999.0)
        thevals['leadingElePhi'] = ak.fill_none(eles.phi[:,0], -999.0)
        thevals['leadingEleDxy'] = ak.fill_none(eles.dxy[:,0], -999.0)
        thevals['leadingEleDz'] = ak.fill_none(eles.dz[:,0], -999.0)
        thevals['leadingEleCharge'] = ak.fill_none(eles.charge[:,0], -999)

        add_weight_variations(thevals, weights, evtmask)
        add_event_id(
            thevals,
            objs.event,
            objs.lumi,
            objs.run,
            evtmask
        )
        return to_pa_table(thevals)