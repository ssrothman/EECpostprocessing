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
            hist.axis.Integer(0, 2, name='sameCharge', label='Charge match'),
            hist.axis.Integer(-1, 2, name='match', label='Matched to gen jet'),
            storage=hist.storage.Weight()
        )

        themask = jetMask[:]
        recoparts = readers.rRecoJet.parts

        wt_b, _ = ak.broadcast_arrays(wt, recoparts.pt)

        sameCharge = recoparts.charge == recoparts.matchCharge
        H.fill(
            pt = squash(recoparts.pt[themask]),
            eta = np.abs(squash(recoparts.eta[themask])),
            pdgid = squash(recoparts.pdgid[themask]),
            match = squash(recoparts.nMatches[themask]),
            matchType = squash(recoparts.matchTypes[themask]),
            sameCharge = squash(sameCharge[themask]),
            weight = squash(wt_b[themask])
        )

        return H

    def genPartMatchHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(6, 0, 3,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.Integer(0, 2, name='match', label='Matched to reco jet'),
            storage=hist.storage.Weight()
        )

        
        tBK = readers.rGenJet._x.ChargedEECsTransferBK
        iReco = tBK.iReco
        iGen = tBK.iGen

        themask = jetMask[iReco]
        genparts = readers.rGenJet.parts[iGen]

        wt_b, _ = ak.broadcast_arrays(wt, genparts.pt)
        
        H.fill(
            pt = squash(genparts.pt[themask]),
            eta = np.abs(squash(genparts.eta[themask])),
            pdgid = squash(genparts.pdgid[themask]),
            match = squash(genparts.nMatches[themask]),
            weight = squash(wt_b[themask])
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

    def genJetMatchHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Jet pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Jet eta'),
            hist.axis.Integer(0, 2, name='match', label='Matched to reco jet'),
            storage=hist.storage.Weight()
        )

        wt_b, _ = ak.broadcast_arrays(wt, readers.rGenJet.jets.pt)

        H.fill(
            pt = squash(readers.rGenJet.jets.pt),
            eta = np.abs(squash(readers.rGenJet.jets.eta)),
            match = squash(readers.rGenJet.simonjets.genJetMatched),
            weight = squash(wt_b)
        )

        return H

    def binAll(self, readers, mask, evtMask, wt):
        HPmatch = self.partMatchHist(readers, mask, wt)
        HJmatch = self.jetMatchHist(readers, mask, wt)

        HgenJmatch = self.genJetMatchHist(readers, mask, wt)
        HgenPmatch = self.genPartMatchHist(readers, mask, wt)

        return {
            'Pmatch' : HPmatch,
            'Jmatch' : HJmatch,
            'genJmatch' : HgenJmatch,
            'genPmatch' : HgenPmatch
        }
