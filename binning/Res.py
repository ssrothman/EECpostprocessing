import hist
import awkward as ak
import numpy as np
from .util import squash

class ResBinner:
    def __init__(self, config, *args, **kwargs):

        self.config = config

    def partResHists(self, readers, jetMask, wt):
        recoparts = readers.rRecoJet.parts

        themask = jetMask & (recoparts.nMatches > 0)

        wt_b, _ = ak.broadcast_arrays(wt, recoparts.pt)
        
        sameCharge = recoparts.charge == recoparts.matchCharge

        #binning variables
        dpt = (recoparts.pt - recoparts.matchPt) / recoparts.matchPt
        deta = recoparts.eta - recoparts.matchEta
        dphi2 = recoparts.phi - recoparts.matchPhi
        dphi1 = ak.where(dphi2 > np.pi, dphi2 - 2*np.pi, dphi2)
        dphi = ak.where(dphi1 < -np.pi, dphi1 + 2*np.pi, dphi1)
        dR = np.sqrt(np.square(dphi) + np.square(deta))

        dpt = squash(dpt[themask])
        deta = squash(deta[themask])
        dphi = squash(dphi[themask])
        dR = squash(dR[themask])
        pt = squash(recoparts.pt[themask])
        eta = np.abs(squash(recoparts.eta[themask]))
        pdgid = squash(recoparts.pdgid[themask])
        matchType = squash(recoparts.matchTypes[themask])
        sameCharge = squash(sameCharge[themask])
        wt_b = squash(wt_b[themask])

        Hpt = hist.Hist(
            hist.axis.Regular(1001, -0.6, 0.6,
                              name='dpt', label="(Reco pT - Gen pT)/Gen pT"),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(3, 0, 2.5,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            hist.axis.Integer(0, 2, name='sameCharge', label='Charge match', overflow=False, underflow=False),
            storage=hist.storage.Weight()
        )

        Hpt.fill(
            dpt = dpt,
            pt = pt,
            eta = eta,
            pdgid = pdgid,
            matchType = matchType,
            sameCharge = sameCharge,
            weight = wt_b   
        )

        Heta = hist.Hist(
            hist.axis.Regular(1001, -0.1, 0.1,
                              name='deta', label='Reco eta - Gen eta'),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(3, 0, 2.5,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            hist.axis.Integer(0, 2, name='sameCharge', label='Charge match', overflow=False, underflow=False),
            storage=hist.storage.Weight()
        )

        Heta.fill(
            deta = deta,
            pt = pt,
            eta = eta,
            pdgid = pdgid,
            matchType = matchType,
            sameCharge = sameCharge,
            weight = wt_b
        )

        Hphi = hist.Hist(
            hist.axis.Regular(1001, -0.1, 0.1,
                              name='dphi', label='Reco phi - Gen phi'),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(3, 0, 2.5,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            hist.axis.Integer(0, 2, name='sameCharge', label='Charge match', overflow=False, underflow=False),
            storage=hist.storage.Weight()
        )


        Hphi.fill(
            dphi = dphi,
            pt = pt,
            eta = eta,
            pdgid = pdgid,
            matchType = matchType,
            sameCharge = sameCharge,
            weight = wt_b
        )

        HdR = hist.Hist(
            hist.axis.Regular(1001, 0.0, 0.1,
                              name='dR', label='deltaR(reco, gen)'),
            hist.axis.Regular(100, 0.1, 100,
                              name='pt', label='Particle pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(3, 0, 2.5,
                              name='eta', label='Particle eta'),
            hist.axis.IntCategory([], name='pdgid', label='Particle type', growth=True),
            hist.axis.IntCategory([], name='matchType', label='Match type', growth=True),
            hist.axis.Integer(0, 2, name='sameCharge', label='Charge match', overflow=False, underflow=False),
            storage=hist.storage.Weight()
        )

        HdR.fill(
            dR = dR,
            pt = pt,
            eta = eta,
            pdgid = pdgid,
            matchType = matchType,
            sameCharge = sameCharge,
        )

        return {
            'pt' : Hpt,
            'eta' : Heta,
            'phi' : Hphi,
            "R" : HdR
        }

    def jetResHists(self, readers, jetMask, wt):
        recojets = readers.rRecoJet.jets
        simonjets = readers.rRecoJet.simonjets

        wt_b, _ = ak.broadcast_arrays(wt, recojets.pt)
        themask = jetMask & (simonjets.jetMatched == 1)

        #binning variables
        dpt_corr = (recojets.corrpt - simonjets.jetMatchPt) / simonjets.jetMatchPt
        dpt_raw = (recojets.pt_raw - simonjets.jetMatchPt) / simonjets.jetMatchPt
        dpt_CMSSW = (recojets.CMSSWpt - simonjets.jetMatchPt) / simonjets.jetMatchPt
        deta = recojets.eta - simonjets.jetMatchEta
        dphi2 = recojets.phi - simonjets.jetMatchPhi
        dphi1 = ak.where(dphi2 > np.pi, dphi2 - 2*np.pi, dphi2)
        dphi = ak.where(dphi1 < -np.pi, dphi1 + 2*np.pi, dphi1)
        dR = np.sqrt(np.square(dphi) + np.square(deta))

        dpt_corr = squash(dpt_corr[themask])
        dpt_raw = squash(dpt_raw[themask])
        dpt_CMSSW = squash(dpt_CMSSW[themask])

        deta = squash(deta[themask])
        dphi = squash(dphi[themask])
        dR = squash(dR[themask])
        pt = squash(simonjets.jetMatchPt[themask])
        eta = np.abs(squash(simonjets.jetMatchEta[themask]))
        wt_b = squash(wt_b[themask])

        Hpt = hist.Hist(
            hist.axis.Regular(201, -1, 1,
                             name='dpt', label="(Reco pT - Gen pT)/Gen pT"),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Gen pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Gen eta'),
            hist.axis.Integer(0, 3, name='type', label='pT type', overflow=False, underflow=False),
            storage=hist.storage.Weight()
        )

        Hpt.fill(
            dpt = dpt_corr,
            pt = pt,
            eta = eta,
            type = 0,
            weight = wt_b
        )
        Hpt.fill(
            dpt = dpt_raw,
            pt = pt,
            eta = eta,
            type = 1,
            weight = wt_b
        )
        Hpt.fill(
            dpt = dpt_CMSSW,
            pt = pt,
            eta = eta,
            type = 2,
            weight = wt_b
        )

        Heta = hist.Hist(
            hist.axis.Regular(201, -1, 1,
                              name='deta', label='Reco eta - Gen eta'),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Gen pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Gen eta'),
            storage=hist.storage.Weight()
        )
        
        Heta.fill(
            deta = deta,
            pt = pt,
            eta = eta,
            weight = wt_b
        )

        Hphi = hist.Hist(
            hist.axis.Regular(201, -1, 1,
                              name='dphi', label='Reco phi - Gen phi'),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Gen pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Gen eta'),
            storage=hist.storage.Weight()
        )


        Hphi.fill(
            dphi = dphi,
            pt = pt,
            eta = eta,
            weight = wt_b
        )

        HdR = hist.Hist(
            hist.axis.Regular(201, 0, 1,
                              name='dR', label='deltaR(reco, gen)'),
            hist.axis.Regular(100, 30, 1200,
                              name='pt', label='Gen pT',
                              transform=hist.axis.transform.log),
            hist.axis.Regular(5, 0, 1.7,
                              name='eta', label='Gen eta'),
            storage=hist.storage.Weight()
        )

        HdR.fill(
            dR = dR,
            pt = pt,
            eta = eta,
            weight = wt_b
        )

        return {
            'pt' : Hpt,
            'eta' : Heta,
            'phi' : Hphi,
            'R' : HdR
        }

    def binAll(self, readers, mask, evtMask, wt):
        HPres = self.partResHists(readers, mask, wt)
        HJres = self.jetResHists(readers, mask, wt)

        return {
            'Pres' : HPres,
            'Jres' : HJres,
        }
