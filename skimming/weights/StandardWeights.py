import os
from typing import Any
import awkward as ak
from correctionlib import CorrectionSet
from skimming.weights.btagSF import single_wp_btagSF
from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import Weights
import numpy as np
from skimming.weights.muonSF import getMuonSF
from skimming.weights.theorySF import getPDFweights, getPSWts, getScaleWts3pt, getScaleWts7pt

class StandardWeights:
    def __init__(self, cfg : dict, evtselcfg : dict):
        self._cfg = cfg
        self._evtselcfg = evtselcfg

    def get_weights(self, allobjects : AllObjects) -> Weights:

        wts = Weights(len(allobjects.event), storeIndividual=True)

        if not allobjects.isMC:
            return wts #short circuit for data

        self._get_theorySFs(wts, allobjects)
        self._get_muonSFs(wts, allobjects)
        self._get_PUSF(wts, allobjects)
        self._get_ZkinSF(wts, allobjects)
        self._get_BtagSF(wts, allobjects)
        self._get_flavorweights(wts, allobjects)
        
        return wts
    
    def _get_theorySFs(self, weights : Weights, allobjects : AllObjects):
        weights.add('generator', allobjects.genWeight)

        getScaleWts7pt(weights, allobjects)
        getScaleWts3pt(weights, allobjects)
        getPDFweights(weights, allobjects)
        getPSWts(weights, allobjects)

    def _get_muonSFs(self, weights : Weights,
                    allobjects : AllObjects):
        
        sfconfig = self._cfg['muonSFs']

        if not sfconfig['skipPrefireSF']:
            prefire = allobjects.PrefireWeight
            weights.add(
                "wt_prefire", 
                prefire.Nom, 
                prefire.Up, 
                prefire.Dn
            )

        muons = allobjects.Muons.muons

        csetpath = sfconfig['path']
        if not csetpath.startswith('/'):
            csetpath = os.path.join(os.path.dirname(__file__), '..', '..', csetpath)
        cset = CorrectionSet.from_file(csetpath)

        mu0 = muons[:,0]
        mu1 = muons[:,1]
        leadmu : Any = ak.where(
            mu0.pt >= mu1.pt,
            mu0,
            mu1
        )
        submu : Any = ak.where(
            mu0.pt < mu1.pt,    
            mu0,
            mu1
        )
        
        leadpt = leadmu.rawPt
        subpt = submu.rawPt

        leadeta = np.abs(leadmu.eta)
        subeta = np.abs(submu.eta)

        whichID = self._evtselcfg['muons']['ID']
        whichIso = self._evtselcfg['muons']['iso']
        whichTrg = self._evtselcfg['global']['trigger']

        #this is the only time it's awkward to not be using a regular dict
        idsfname = sfconfig['idsfnames'][whichID]
        isosfname = sfconfig['isosfnames'][whichIso][whichID]
        triggersfname = sfconfig['triggersfnames'][whichTrg]

        if not sfconfig['skipIDsf']:
            idsf_lead, idsf_lead_up, idsf_lead_dn = getMuonSF(
                cset, 
                idsfname, 
                leadeta, 
                leadpt
            )

            idsf_sub, idsf_sub_up, idsf_sub_dn = getMuonSF(
                cset, 
                idsfname, 
                subeta, 
                subpt
            )

            weights.add(
                "wt_idsf", 
                idsf_lead*idsf_sub, 
                idsf_lead_up*idsf_sub_up, 
                idsf_lead_dn*idsf_sub_dn
            )
            
        if not sfconfig['skipIsosf']:
            isosf_lead, isosf_lead_up, isosf_lead_dn = getMuonSF(
                cset, 
                isosfname, 
                leadeta,
                leadpt
            )

            isosf_sub, isosf_sub_up, isosf_sub_dn = getMuonSF(
                cset,
                isosfname,
                subeta, 
                subpt
            )

            weights.add(
                "wt_isosf", 
                isosf_lead*isosf_sub, 
                isosf_lead_up*isosf_sub_up, 
                isosf_lead_dn*isosf_sub_dn
            )
            
        if not sfconfig['skipTriggersf']:
            triggersf, triggersf_up, triggersf_dn = getMuonSF(cset, 
                                                        triggersfname, 
                                                        leadeta, 
                                                        leadpt)

            weights.add(
                "wt_triggersf",
                triggersf, 
                triggersf_up,
                triggersf_dn
            )

    def _get_PUSF(self, weights : Weights,
                    allobjects : AllObjects):
        
        PUcfg = self._cfg['PUreweight']

        if PUcfg['skip']:
            return
        
        csetpath = PUcfg['path']
        if not csetpath.startswith('/'):
            csetpath = os.path.join(os.path.dirname(__file__), '..', '..', csetpath)
        cset = CorrectionSet.from_file(csetpath)
        ev = cset[PUcfg['name']]

        nTruePU = allobjects.PileupInfo.nTrueInt

        nom = ev.evaluate(
            nTruePU,
            "nominal"
        )

        up = ev.evaluate(
            nTruePU,
            "up"
        )

        dn = ev.evaluate(
            nTruePU,
            "down"
        )

        weights.add("wt_PU", nom, up, dn)
    
    def _get_ZkinSF(self, weights : Weights,
                    allobjects : AllObjects):
        
        Zcfg = self._cfg['Zreweight']

        if Zcfg['skip']:
            return
        
        csetpath = Zcfg['path']
        if not csetpath.startswith('/'):
            csetpath = os.path.join(os.path.dirname(__file__), '..', '..', csetpath)
        cset = CorrectionSet.from_file(csetpath)

        Zs = allobjects.Zs.Zs

        badpt = ak.is_none(Zs.pt)
        Zpt = ak.fill_none(Zs.pt, 0)

        bady = ak.is_none(Zs.rapidity) | (np.abs(Zs.rapidity) > 2.4)
        Zy = ak.fill_none(Zs.rapidity, 0)
        
        Zsf = cset[Zcfg['name']].evaluate(Zpt, np.abs(Zy))

        Zsf = np.where(badpt | bady, 1, Zsf)
        Zsf = np.where(Zsf <=0, 1, Zsf) #protect against zeros.
                                        #shouldn't happen, but just in case

        weights.add('wt_Zkin', Zsf)

    def _get_BtagSF(self, weights : Weights,
                    allobjects : AllObjects):
         
        btagcfg = self._cfg['btagSF']

        if btagcfg['skip']:
            return
         
        if self._evtselcfg['global']['maxNumBtag'] < 0:
            return

        sfpath= btagcfg['sfpath']
        if not sfpath.startswith('/'):
            sfpath = os.path.join(os.path.dirname(__file__), '..', '..', sfpath)
        efpath= btagcfg['effpath']
        if not efpath.startswith('/'):
            efpath = os.path.join(os.path.dirname(__file__), '..', '..', efpath)
        cset_sf = CorrectionSet.from_file(sfpath)
        cset_eff = CorrectionSet.from_file(efpath)
        
        jets = allobjects.AK4Jets.jets
        
        pt = jets.pt
        abseta = np.abs(jets.eta)
        flav = jets.hadronFlavour

        num = ak.num(jets, axis=1)

        pt = ak.flatten(pt)
        abseta = ak.flatten(abseta)
        flav = ak.flatten(flav)

        vetowp = self._evtselcfg['global']['maxNumBtag_level']
        passwp = jets['pass%sB' % vetowp]
        passwp = ak.flatten(passwp)

        single_wp_btagSF(
            weights,
            pt, 
            abseta, 
            flav, 
            passwp, 
            num, 
            vetowp, 
            cset_sf, 
            cset_eff
        )
   
    def _get_flavorweights(self, weights : Weights,
                           allobjects : AllObjects):
        flavs = [0, 4, 5, 21]
        names = ['uds', 'c', 'b', 'g']

        for flav, name in zip(flavs, names):
            flavormask = ak.any(allobjects.RecoJets.jets.hadronFlavour == flav, axis=-1)
            factor = self._cfg['flavorWeights'][name]
            if factor > 0:
                continue

            w_nom = np.ones(len(flavormask))
            w_up = w_nom[:]
            w_dn = w_nom[:]

            w_up[flavormask] += factor
            w_dn[flavormask] -= factor

            weights.add("wt_%s_xsec"%name, w_nom, w_up, w_dn)