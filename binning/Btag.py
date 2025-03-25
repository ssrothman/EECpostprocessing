import hist
import awkward as ak
import numpy as np
from .util import squash

class BtagBinner:
    def __init__(self, config, *args, **kwargs):
        self.config = config

    def binBtag(self, readers, mask, wt):
        H = hist.Hist(
            hist.axis.Variable(self.config.binning.bins.Btag.pt,
                               name='pt', label='Puppi Jet pT'),
            hist.axis.Variable(self.config.binning.bins.Btag.eta,
                               name='eta', label='Puppi Jet eta'),
            hist.axis.Integer(0, 2,
                              name='looseB', label='Loose b-tagged'),
            hist.axis.Integer(0, 2,
                              name='mediumB', label='Medium b-tagged'),
            hist.axis.Integer(0, 2,
                              name='tightB', label='Tight b-tagged'),
            hist.axis.IntCategory([0, 4, 5, 21],
                                  name='flavor', label='Jet type'),
            storage=hist.storage.Weight()
        )

        wt_b = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)[0]

        H.fill(
            pt = squash(readers.rRecoJet.jets.pt[mask]),
            eta = squash(np.abs(readers.rRecoJet.jets.eta[mask])),
            looseB = squash(readers.rRecoJet.jets.passLooseB[mask]),
            mediumB = squash(readers.rRecoJet.jets.passMediumB[mask]),
            tightB = squash(readers.rRecoJet.jets.passTightB[mask]),
            flavor = squash(readers.rRecoJet.jets.hadronFlavour[mask]),
            weight = squash(wt_b[mask])
        )

        return H

    def binBmatch(self, readers, mask, wt):
        H = hist.Hist(
            hist.axis.Variable(self.config.binning.bins.Btag.pt,
                               name='pt', label='Puppi Jet pT'),
            hist.axis.Variable(self.config.binning.bins.Btag.eta,
                               name='eta', label='Puppi Jet eta'),
            hist.axis.Integer(0, 3,
                              name='nmatch',
                              label='Number of matching CHS jets'),
            storage=hist.storage.Weight()
        )

        wt_b = ak.broadcast_arrays(wt, readers.rRecoJet.jets.pt)[0]
        nmatch = ak.num(readers.rRecoJet.CHSjets.pt[readers.rRecoJet.CHSjets.pt > 0], axis=-1)

        H.fill(
            pt = squash(readers.rRecoJet.jets.pt[mask]),
            eta = squash(np.abs(readers.rRecoJet.jets.eta[mask])),
            nmatch = squash(nmatch[mask]),
            weight = squash(wt_b[mask])
        )

        return H

    def binAll(self, readers, mask, evtMask, wt):
        Hbtag = self.binBtag(readers, mask, wt)
        Hbmatch = self.binBmatch(readers, mask, wt)
        return {
            "Btag" : Hbtag,
            "Bmatch" : Hbmatch
        }
