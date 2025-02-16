import hist
import awkward as ak
import numpy as np
from .util import squash

class MatchBinner:
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit, sepPt):

        if manualcov or poissonbootstrap or statsplit > 1 or sepPt:
            raise ValueError("Invalid configuration for MatchBinner")

        self.config = config

    def partMatchHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(6, 0, 3,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            hist.axis.Integer(-1, 2, name='match', label='Matched to gen jet'),
            storage=hist.storage.Weight()
        )

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.parts.pt)

        H.fill(
            pt = squash(readers.rRecoJet.parts.pt[jetMask]),
            eta = np.abs(squash(readers.rRecoJet.parts.eta[jetMask])),
            pdgid = squash(readers.rRecoJet.parts.pdgid[jetMask]),
            match = squash(readers.rRecoJet.parts.nMatches[jetMask]),
            matchType = squash(readers.rRecoJet.parts.matchTypes[jetMask]),
            weight = squash(wt_b[jetMask])
        )

        return H

    def jetMatchHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Jet pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Jet eta'),
            hist.axis.Integer(0, 2, name='match', label='Matched to gen jet'),
            storage=hist.storage.Weight()
        )

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)

        H.fill(
            pt = squash(readers.rRecoJet.jets.pt[jetMask]),
            eta = np.abs(squash(readers.rRecoJet.jets.eta[jetMask])),
            match = squash(readers.rRecoJet.simonjets.jetMatched[jetMask]),
            weight = squash(wt_b[jetMask])
        )

        return H

    def partResHists(self, readers, jetMask, wt):
        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.parts.pt)
        themask = jetMask & (readers.rRecoJet.parts.nMatches > 0)
        print("Part Res Hists");
        print("min nmatches:", ak.min(readers.rRecoJet.parts.nMatches[themask]))

        Hpt = hist.Hist(
            hist.axis.Regular(201, -1, 1,
                              name='dpt', label="(Reco pT - Gen pT)/Gen pT"),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(6, 0, 3,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            storage=hist.storage.Weight()
        )
        dpt = (readers.rRecoJet.parts.pt - readers.rRecoJet.parts.matchPt) / readers.rRecoJet.parts.matchPt
        print("min dpt:", ak.min(dpt[themask]))
        print("max dpt:", ak.max(dpt[themask]))

        Hpt.fill(
            dpt = squash(dpt[themask]),
            pt = squash(readers.rRecoJet.parts.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.parts.eta[themask])),
            pdgid = squash(readers.rRecoJet.parts.pdgid[themask]),
            matchType = squash(readers.rRecoJet.parts.matchTypes[themask]),
            weight = squash(wt_b[themask])
        )

        Heta = hist.Hist(
            hist.axis.Regular(201, -0.1, 0.1,
                              name='deta', label='Reco eta - Gen eta'),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(6, 0, 3,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            storage=hist.storage.Weight()
        )

        deta = readers.rRecoJet.parts.eta - readers.rRecoJet.parts.matchEta
        print("min deta:", ak.min(deta[themask]))
        print("max deta:", ak.max(deta[themask]))
        Heta.fill(
            deta = squash(deta[themask]),
            pt = squash(readers.rRecoJet.parts.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.parts.eta[themask])),
            pdgid = squash(readers.rRecoJet.parts.pdgid[themask]),
            matchType = squash(readers.rRecoJet.parts.matchTypes[themask]),
            weight = squash(wt_b[themask])
        )

        Hphi = hist.Hist(
            hist.axis.Regular(201, -0.1, 0.1,
                              name='dphi', label='Reco phi - Gen phi'),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(6, 0, 3,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            storage=hist.storage.Weight()
        )

        dphi2 = readers.rRecoJet.parts.phi - readers.rRecoJet.parts.matchPhi
        dphi1 = ak.where(dphi2 > np.pi, dphi2 - 2*np.pi, dphi2)
        dphi = ak.where(dphi1 < -np.pi, dphi1 + 2*np.pi, dphi1)
        print("min dphi:", ak.min(dphi[themask]))
        print("max dphi:", ak.max(dphi[themask]))

        Hphi.fill(
            dphi = squash(dphi[themask]),
            pt = squash(readers.rRecoJet.parts.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.parts.eta[themask])),
            pdgid = squash(readers.rRecoJet.parts.pdgid[themask]),
            matchType = squash(readers.rRecoJet.parts.matchTypes[themask]),
            weight = squash(wt_b[themask])
        )

        return {
            'pt' : Hpt,
            'eta' : Heta,
            'phi' : Hphi
        }

    def jetResHists(self, readers, jetMask, wt):
        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)
        themask = jetMask & (readers.rRecoJet.simonjets.jetMatched == 1)

        Hpt = hist.Hist(
            hist.axis.Regular(101, -1, 1,
                             name='dpt', label="(Reco pT - Gen pT)/Gen pT"),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Reco pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Reco eta'),
            storage=hist.storage.Weight()
        )
        dpt = (readers.rRecoJet.jets.pt - readers.rRecoJet.simonjets.jetMatchPt) / readers.rRecoJet.simonjets.jetMatchPt

        Hpt.fill(
            dpt = squash(dpt[themask]),
            pt = squash(readers.rRecoJet.jets.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.jets.eta[themask])),
            weight = squash(wt_b[themask])
        )

        Heta = hist.Hist(
            hist.axis.Regular(101, -1, 1,
                              name='deta', label='Reco eta - Gen eta'),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Reco pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Reco eta'),
            storage=hist.storage.Weight()
        )
        deta = readers.rRecoJet.jets.eta - readers.rRecoJet.simonjets.jetMatchEta
        
        Heta.fill(
            deta = squash(deta[themask]),
            pt = squash(readers.rRecoJet.jets.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.jets.eta[themask])),
            weight = squash(wt_b[themask])
        )

        Hphi = hist.Hist(
            hist.axis.Regular(101, -1, 1,
                              name='dphi', label='Reco phi - Gen phi'),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Reco pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Reco eta'),
            storage=hist.storage.Weight()
        )

        dphi2 = readers.rRecoJet.jets.phi - readers.rRecoJet.simonjets.jetMatchPhi
        dphi1 = ak.where(dphi2 > np.pi, dphi2 - 2*np.pi, dphi2)
        dphi = ak.where(dphi1 < -np.pi, dphi1 + 2*np.pi, dphi1)

        Hphi.fill(
            dphi = squash(dphi[themask]),
            pt = squash(readers.rRecoJet.jets.pt[themask]),
            eta = np.abs(squash(readers.rRecoJet.jets.eta[themask])),
            weight = squash(wt_b[themask])
        )

        return {
            'pt' : Hpt,
            'eta' : Heta,
            'phi' : Hphi
        }

    def binAll(self, readers, mask, evtMask, wt):
        HPmatch = self.partMatchHist(readers, mask, wt)
        HJmatch = self.jetMatchHist(readers, mask, wt)

        HPres = self.partResHists(readers, mask, wt)
        HJres = self.jetResHists(readers, mask, wt)

        return {
            'Pmatch' : HPmatch,
            'Jmatch' : HJmatch,
            'Pres' : HPres,
            'Jres' : HJres
        }
