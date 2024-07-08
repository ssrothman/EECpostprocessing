import hist
import awkward as ak
import numpy as np
from .util import squash

class MatchBinner:
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit, sepPt):

        if manualcov or poissonbootstrap or statsplit or sepPt:
            raise ValueError("Invalid configuration for MatchBinner")

        self.config = config

    def partMatchHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.match.partpt,
                              name='pt', label='Particle pT'),
            hist.axis.Regular(*self.config.binning.bins.match.parteta,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([11, 13, 22, 130, 211],
                                  name='pdgid', label='Particle type'),
            hist.axis.Integer(0, 2,
                              name='match', label='Matched to gen jet'),
            storage=hist.storage.Weight()
        )

        matched = readers.rRecoJet.parts.nmatch > 0

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.parts.pt)

        H.fill(
            pt = squash(readers.rRecoJet.parts.pt[jetMask]),
            eta = np.abs(squash(readers.rRecoJet.parts.eta[jetMask])),
            pdgid = squash(readers.rRecoJet.parts.pdgid[jetMask]),
            match = squash(matched[jetMask]),
            weight = squash(wt_b[jetMask])
        )

        return H

    def partResHist(self, readers, jetMask, wt, pdgid):
        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.match.partpt,
                              name='pt', label='Particle pT'),
            hist.axis.Regular(
                *vars(self.config.binning.bins.match)[str(pdgid)].dpt,
                name='dpt', label='Particle pT match'),
            hist.axis.Regular(
                *vars(self.config.binning.bins.match)[str(pdgid)].deta,
                name='deta', label='Particle eta match'),
            hist.axis.Regular(
                *vars(self.config.binning.bins.match)[str(pdgid)].dphi,
                name='dphi', label='Particle phi match'),
            storage=hist.storage.Weight()
        )

        dpt = (readers.rRecoJet.parts.matchPt - readers.rRecoJet.parts.pt)/readers.rRecoJet.parts.matchPt
        dphi = (readers.rRecoJet.parts.matchPhi - readers.rRecoJet.parts.phi)
        deta = (readers.rRecoJet.parts.matchEta - readers.rRecoJet.parts.eta)

        mask = jetMask & (readers.rRecoJet.parts.nmatch > 0) & (readers.rRecoJet.parts.pdgid == pdgid)

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.parts.pt)

        H.fill(
            pt = squash(readers.rRecoJet.parts.pt[mask]),
            dpt = squash(dpt[mask]),
            deta = squash(deta[mask]),
            dphi = squash(dphi[mask]),
            weight = squash(wt_b[mask])
        )

        return H


    def jetMatchHist(self, readers, jetMask, wt):

        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.match.Jpt,
                              name='pt', label='Jet pT'),
            hist.axis.Regular(*self.config.binning.bins.match.Jeta,
                              name='eta', label='Jet eta'),
            hist.axis.Integer(0, 2,
                              name='match', label='Matched to gen jet'),
            storage=hist.storage.Weight()
        )

        matched = readers.rRecoJet.simonjets.genPt < 0
        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)

        H.fill(
            pt = squash(readers.rRecoJet.jets.pt[jetMask]),
            eta = np.abs(squash(readers.rRecoJet.jets.eta[jetMask])),
            match = squash(matched[jetMask]),
            weight = squash(wt_b[jetMask])
        )

        return H

    def jetResolutionHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.match.Jpt,
                              name='pt', label='Jet pT'),
            hist.axis.Regular(*self.config.binning.bins.match.Jdpt,
                              name='dpt', label='Jet pT match'),
            hist.axis.Regular(*self.config.binning.bins.match.Jdeta,
                              name='deta', label='Jet eta match'),
            hist.axis.Regular(*self.config.binning.bins.match.Jdphi,
                              name='dphi', label='Jet phi match'),
            storage=hist.storage.Weight()
        )

        dpt = (readers.rRecoJet.simonjets.genPt - readers.rRecoJet.jets.pt)/readers.rRecoJet.simonjets.genPt
        dphi = (readers.rRecoJet.simonjets.genPhi - readers.rRecoJet.jets.phi)
        deta = (readers.rRecoJet.simonjets.genEta - readers.rRecoJet.jets.eta)

        mask = jetMask & (readers.rRecoJet.simonjets.genPt >= 0)

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)

        H.fill(
            pt = squash(readers.rRecoJet.jets.pt[mask]),
            dpt = squash(dpt[mask]),
            deta = squash(deta[mask]),
            dphi = squash(dphi[mask]),
            weight = squash(wt_b[mask])
        )

        return H

    def binAll(self, readers, mask, evtMask, wt):
        Hjmatch = self.jetMatchHist(readers, mask, wt)
        Hjres = self.jetResolutionHist(readers, mask, wt)
    
        Hpmatch = self.partMatchHist(readers, mask, wt)
        Hpres = {}
        pdgnames = {11 : 'ELE', 
                    13 : 'MU', 
                    22 : 'EM0', 
                    130 : 'HAD0', 
                    211 : 'HADCH'}
        for pdgid in [11, 13, 22, 130, 211]:
            Hpres[pdgnames[pdgid]] = self.partResHist(readers, mask, wt, pdgid)

        return {
            "Jmatch" : Hjmatch,
            "Jres" : Hjres,
            'Pmatch' : Hpmatch,
            'Pres' : Hpres
        }
