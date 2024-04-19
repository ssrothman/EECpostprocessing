import numpy as np
import awkward as ak
from hist.storage import Double, Weight
from hist import Hist
import hist

from .util import *

class MatchBinner:
    def __init__(self, config, config_tag):
        self._config = {}
        self._config['axes'] = config.axes
        self._config['bins'] = vars(config.bins)
        self._config['skipTrans'] = vars(config.skipTransfer)
        self._config['diagTrans'] = vars(config.diagTransfer)

        self._config['tag'] = vars(config_tag)

    def getResHists(self, name):
        dpt = hist.Hist(
            getAxis("%sdpt"%name, self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("partPt", self._config['bins']),
            storage=Weight()
        )
        deta = hist.Hist(
            getAxis("%sdeta"%name, self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("partPt", self._config['bins']),
            storage=Weight()
        )
        dphi = hist.Hist(
            getAxis("%sdphi"%name, self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("partPt", self._config['bins']),
            storage=Weight()
        )
        return dpt, deta, dphi

    def makeAndFillPartRes(self, rJet, jetMask, weight, name):
        Hdpt, Hdeta, Hdphi = self.getResHists(name)

        if hasattr(rJet.jets, 'corrpt'):
            Jpt = rJet.jets.corrpt[jetMask]
        else:
            Jpt = rJet.jets.pt[jetMask]

        if name == 'EM0':
            partMask = rJet.parts.pdgid[jetMask] == 22
        elif name == 'HAD0':
            partMask = rJet.parts.pdgid[jetMask] == 130
        elif name == 'TRK':
            partMask = rJet.parts.charge[jetMask] != 0
            
        Ppt = rJet.parts.pt[jetMask][partMask]
        Peta = rJet.parts.eta[jetMask][partMask]
        Pphi = rJet.parts.phi[jetMask][partMask]
    
        Mpt = rJet.parts.matchPt[jetMask][partMask]
        Meta = rJet.parts.matchEta[jetMask][partMask]
        Mphi = rJet.parts.matchPhi[jetMask][partMask]

        dpt = (Ppt - Mpt)/Mpt
        deta = Peta - Meta
        dphi = Pphi - Mphi

        Jpt, weight, _ = ak.broadcast_arrays(Jpt, weight, dpt)

        Hdpt.fill(
            dpt=squash(dpt),
            eta=squash(Peta),
            partPt=squash(Ppt),
            weight=squash(weight)
        )
        Hdeta.fill(
            deta=squash(deta),
            eta=squash(Peta),
            partPt=squash(Ppt),
            weight=squash(weight)
        )
        Hdphi.fill(
            dphi=squash(dphi),
            eta=squash(Peta),
            partPt=squash(Ppt),
            weight=squash(weight)
        )

        return Hdpt, Hdeta, Hdphi

    def makeAndFillJetRes(self, rJet, jetMask, weight):
        Hdpt, Hdeta, Hdphi = self.getResHists("J")

        extramask = rJet.jets.pt_gen[jetMask] > 0

        if hasattr(rJet.jets, 'corrpt'):
            Jpt = rJet.jets.corrpt[jetMask][extramask]
        else:
            Jpt = rJet.jets.pt[jetMask][extramask]

        Jeta = rJet.jets.eta[jetMask][extramask]
        Jphi = rJet.jets.phi[jetMask][extramask]

        Gpt = rJet.jets.pt_gen[jetMask][extramask]
        Geta = rJet.jets.eta_gen[jetMask][extramask]
        Gphi = rJet.jets.phi_gen[jetMask][extramask]

        dpt = (Jpt - Gpt)/Gpt
        deta = Jeta - Geta
        dphi = Jphi - Gphi

        weight, _ = ak.broadcast_arrays(weight, dpt)

        print(squash(dpt).shape)
        print(squash(Jeta).shape)
        print(squash(Gpt).shape)
        print(squash(weight).shape)

        Hdpt.fill(
            dpt=squash(dpt),
            eta=squash(Jeta),
            partPt=squash(Gpt),
            weight=squash(weight)
        )
        Hdeta.fill(
            deta=squash(deta),
            eta=squash(Jeta),
            partPt=squash(Gpt),
            weight=squash(weight)
        )
        Hdphi.fill(
            dphi=squash(dphi),
            eta=squash(Jeta),
            partPt=squash(Gpt),
            weight=squash(weight)
        )

        return Hdpt, Hdeta, Hdphi
        
    def getPartMatchHist(self):
        return hist.Hist(
            getAxis("partPt", self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("Jpt", self._config['bins']),
            getAxis("DRaxis", self._config['bins']),
            getAxis("partSpecies", self._config['bins']),
            getAxis("nMatch", self._config['bins']),
            getAxis("btag", self._config['bins']),
            storage=Weight()
        )

    def makeAndFillPartMatch(self, rJet, idx, jetMask, weight):
        H = self.getPartMatchHist()

        if hasattr(rJet.jets, 'corrpt'):
            Jpt = rJet.jets.corrpt[idx][jetMask[idx]]
        else:
            Jpt = rJet.jets.pt[idx][jetMask[idx]]
        Jeta = rJet.jets.eta[idx][jetMask[idx]]
        Jphi = rJet.jets.phi[idx][jetMask[idx]]

        Ppt = rJet.parts.pt[idx][jetMask[idx]]
        Peta = rJet.parts.eta[idx][jetMask[idx]]
        Pphi = rJet.parts.phi[idx][jetMask[idx]]

        deta = np.abs(Jeta - Peta)
        dphi = np.abs(Jphi - Pphi)
        dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        dphi = np.where(dphi < -np.pi, 2*np.pi + dphi, dphi)
        dR = np.sqrt(deta*deta + dphi*dphi)

        partSpecies = ak.where(rJet.parts.charge[idx][jetMask[idx]] != 0, 0,
                   ak.where(rJet.parts.pdgid[idx][jetMask[idx]] == 22, 1, 2))

        nMatch = rJet.parts.nmatch[idx][jetMask[idx]]

        btag = getTag(rJet, idx, jetMask, self._config['tag'])

        Jpt, btag, weight, _ = ak.broadcast_arrays(Jpt, btag, weight, Ppt)

        print("partMatch")
        print(squash(Ppt).shape)
        print(squash(Peta).shape)
        print(squash(Jpt).shape)
        print(squash(dR).shape)
        print(squash(partSpecies).shape)
        print(squash(nMatch).shape)
        print(squash(btag).shape)
        print(squash(weight).shape)
        print()


        H.fill(
            partPt=squash(Ppt),
            eta=squash(Peta),
            Jpt=squash(Jpt),
            DRaxis=squash(dR),
            partSpecies=squash(partSpecies),
            nMatch=squash(nMatch),
            btag=squash(btag),
            weight=squash(weight)
        )

        return H

    def getJetMatchHist(self):
        return hist.Hist(
            getAxis("Jpt", self._config['bins']),
            getAxis("eta", self._config['bins']),
            getAxis("btag", self._config['bins']),
            getAxis("hasMatch", self._config['bins']),
            getAxis("fracMatched", self._config['bins']),
            storage=Weight()
        )

    def makeAndFillJetMatch(self, rJet, jetMask, weight):
        H = self.getJetMatchHist()

        if hasattr(rJet.jets, 'corrpt'):
            Jpt = rJet.jets.corrpt[jetMask]
        else:
            Jpt = rJet.jets.pt[jetMask]
        Jeta = rJet.jets.eta[jetMask]

        btag = getTag(rJet, slice(None), jetMask, self._config['tag'])

        hasMatch = rJet.jets.pt_gen[jetMask] > 0

        matched = rJet.parts.nmatch[jetMask] > 0
        matchedPt = rJet.parts.pt[jetMask][matched]
        summatchedPt = ak.sum(matchedPt, axis=-1)
        sumpt = ak.sum(rJet.parts.pt[jetMask], axis=-1)
        fracMatched = summatchedPt/sumpt

        weight = ak.broadcast_arrays(weight, Jpt)[0]
        print("JetMatch")
        print(squash(Jpt).shape)
        print(squash(Jeta).shape)
        print(squash(btag).shape)
        print(squash(hasMatch).shape)
        print(squash(fracMatched).shape)
        print(squash(weight).shape)

        H.fill(
            Jpt=squash(Jpt),
            eta=squash(Jeta),
            btag=squash(btag),
            hasMatch=squash(hasMatch),
            fracMatched=squash(fracMatched),
            weight=squash(weight)
        )

        return H

    def binAll(self, readers, jetMask, evtMask, wt):
        resolutions = {}
        for name in ['EM0', 'HAD0', 'TRK']:
            Hdpt, Hdeta, Hdphi = self.makeAndFillPartRes(readers.rRecoJet, 
                                                         jetMask, wt, name)
            resolutions[name] = {
                'dpt': Hdpt,
                'deta': Hdeta,
                'dphi': Hdphi,
            }

        Hdpt, Hdeta, Hdphi = self.makeAndFillJetRes(readers.rRecoJet, jetMask, wt)
        resolutions['JET'] = {
            'dpt': Hdpt,
            'deta': Hdeta,
            'dphi': Hdphi,
        }

        HPmatch = self.makeAndFillPartMatch(readers.rRecoJet, 
                                            slice(None), 
                                            jetMask, wt)
        #HPmatchGen = self.makeAndFillPartMatch(readers.rGenJet, 
        #                                       readers.rMatch.iGen,
        #                                       jetMask, wt)

        HJmatch = self.makeAndFillJetMatch(readers.rRecoJet, jetMask, wt)

        return {
            'res' : resolutions,
            'partmatch' : HPmatch,
        #    'genpartmatch' : HPmatchGen,
            'jetmatch' : HJmatch,
        }

