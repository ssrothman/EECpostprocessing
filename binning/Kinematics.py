import hist
import awkward as ak
import numpy as np
from .util import squash

class KinematicsBinner:
    def __init__(self, config,
                 manualcov, poissonbootstrap, statsplit, sepPt):
            
        if manualcov or poissonbootstrap or statsplit > 1 or sepPt:
            raise ValueError("Invalid configuration for KinematicsBinner")

        self.config = config

    def makeGenHTHist(self, readers, evtMask, wt):
        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.HT,
                              name='HT', label='Gen HT'),
            storage=hist.storage.Double()
        )

        if readers.LHE is not None:
            H.fill(
                HT = squash(readers.LHE.HT)
            )

        return H

    def makePUHist(self, readers, evtMask, wt):
        H = hist.Hist(
            hist.axis.Integer(0, self.config.binning.bins.nTrueInt,
                             name='nTrueInt', 
                             label='Number of true interactions'),
            hist.axis.Regular(*self.config.binning.bins.rho,
                              name='rho', label='rho'),
            storage=hist.storage.Weight()
        )

        if readers.nTrueInt is not None:
            H.fill(
                nTrueInt = squash(readers.nTrueInt[evtMask]).astype(np.int32),
                rho      = squash(readers.rho[evtMask]),
                weight   = squash(wt[evtMask])
            )
        else:
            H.fill(
                nTrueInt = np.zeros_like(evtMask).astype(np.int32)[evtMask],
                rho      = squash(readers.rho[evtMask]),
                weight   = squash(wt[evtMask])
            )

        return H
        

    def makeSelvarHist(self, readers, evtMask, wt):
        H = hist.Hist(
            hist.axis.Regular(*self.config.binning.bins.MET,
                              name='MET', label='MET pT'),
            hist.axis.Integer(0, 3,
                              name='numLooseB', 
                              label='Number of b-tagged jets'),
            hist.axis.Integer(0, 3,
                              name='numMediumB', 
                              label='Number of b-tagged jets'),
            hist.axis.Integer(0, 3,
                              name='numTightB', 
                              label='Number of b-tagged jets'),
            storage=hist.storage.Weight()
        )

        numLooseB = ak.sum(readers.rRecoJet.jets.passLooseB, axis=-1)
        numMediumB = ak.sum(readers.rRecoJet.jets.passMediumB, axis=-1)
        numTightB = ak.sum(readers.rRecoJet.jets.passTightB, axis=-1)

        H.fill(
            MET        = squash(readers.METpt[evtMask]),
            numLooseB  = squash(numLooseB[evtMask]),
            numMediumB = squash(numMediumB[evtMask]),
            numTightB  = squash(numTightB[evtMask]),
            weight     = squash(wt[evtMask])
        )

        return H

    def makeJetHist(self, readers, jetMask, wt):
        H = hist.Hist(
            hist.axis.Variable(self.config.binning.bins.Jpt,
                               name='pt', label='Jet pT'),
            hist.axis.Regular(*self.config.binning.bins.Jeta,
                              name='eta', label='Jet eta'),
            hist.axis.Integer(0, 2, 
                              name='passTightB', label='Pass tight b-tag'),
            hist.axis.Integer(0, 2, 
                              name='passMediumB', label='Pass medium b-tag'),
            hist.axis.Integer(0, 2, 
                              name='passLooseB', label='Pass loose b-tag'),
            storage=hist.storage.Weight()
        )

        wt_b, _ = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)

        H.fill(
            pt     = squash(readers.rRecoJet.jets.pt[jetMask]),
            eta    = np.abs(squash(readers.rRecoJet.jets.eta[jetMask])),
            passLooseB  = squash(readers.rRecoJet.jets.passLooseB[jetMask]),
            passMediumB  = squash(readers.rRecoJet.jets.passMediumB[jetMask]),
            passTightB  = squash(readers.rRecoJet.jets.passTightB[jetMask]),
            weight = squash(wt_b[jetMask])
        )

        return H


    def getMuonHist(self):
        return hist.Hist(
            hist.axis.Variable(self.config.binning.bins.MUpt,
                               name='pt', label='Muon pT'),
            hist.axis.Regular(*self.config.binning.bins.MUeta,
                              name='eta', label='Muon eta'),
            storage=hist.storage.Weight()
        )

    def makeZHist(self, readers, evtMask, wt):
        H = hist.Hist(
            hist.axis.Variable(self.config.binning.bins.Zpt,
                               name='pt', label='Z pt'),
            hist.axis.Regular(*self.config.binning.bins.Zy,
                              name='y', label='Z y'),
            hist.axis.Regular(*self.config.binning.bins.Zmass,
                              name='mass', label='Z mass'),
            storage=hist.storage.Weight() 
        )

        Z = readers.rMu.Zs
        H.fill(
            pt     = squash(Z.pt[evtMask]),
            y      = np.abs(squash(Z.y[evtMask])),
            mass   = squash(Z.mass[evtMask]),
            weight = squash(wt[evtMask])
        )

        return H

    def makeMuonHists(self, readers, evtMask, wt):
        Hlead = self.getMuonHist()
        Hsub = self.getMuonHist()

        mu0 = readers.rMu.muons[:,0]
        mu1 = readers.rMu.muons[:,1]

        leadmu = ak.where(mu0.pt > mu1.pt, mu0, mu1)
        submu = ak.where(mu0.pt > mu1.pt, mu1, mu0)

        Hlead.fill(
            pt     = squash(leadmu.pt[evtMask]), 
            eta    = np.abs(squash(leadmu.eta[evtMask])), 
            weight = squash(wt[evtMask])
        )

        Hsub.fill(
            pt     = squash(submu.pt[evtMask]),
            eta    = np.abs(squash(submu.eta[evtMask])),
            weight = squash(wt[evtMask])
        )

        return Hlead, Hsub
        
    def binAll(self, readers, mask, evtMask, wt):
        HleadMu, HsubMu = self.makeMuonHists(readers, evtMask, wt)
        HZ = self.makeZHist(readers, evtMask, wt)
        HHT = self.makeGenHTHist(readers, evtMask, wt)
        HPU = self.makePUHist(readers, evtMask, wt)
        Hselvar = self.makeSelvarHist(readers, evtMask, wt)
        Hjet = self.makeJetHist(readers, mask, wt)

        return {
            'leadMu': HleadMu,
            'subMu': HsubMu,
            'Z': HZ,
            'jets': Hjet,
            'HT': HHT,
            'PU': HPU,
            'selvar': Hselvar,
        }
