import hist
import awkward as ak
import numpy as np
from .util import squash

class BeffBinner:
    def __init__(self, config, *args, **kwargs):
        self.config = config

    def binAll(self, readers, mask, evtMask, wt):
        H = hist.Hist(
            hist.axis.Variable(self.config.binning.bins.Beff.pt,
                              name='pt', label='CHS Jet pT'),
            hist.axis.Variable(self.config.binning.bins.Beff.eta,
                              name='eta', label='CHS Jet eta'),
            hist.axis.Integer(0, 2,
                              name='looseB', label='Loose b-tagged'),
            hist.axis.Integer(0, 2,
                              name='mediumB', label='Medium b-tagged'),
            hist.axis.Integer(0, 2,
                              name='tightB', label='Tight b-tagged'),
            hist.axis.IntCategory([0, 4, 5],
                                  name='flavor', label='Jet flavor'),
            storage=hist.storage.Weight()
        )

        jets = readers.rRecoJet.CHSjets
    
        wt_b = ak.broadcast_arrays(wt, jets.pt)[0]

        print(np.unique(squash(jets.hadronFlavour)))

        H.fill(
            pt = squash(jets.pt[mask]),
            eta = squash(np.abs(jets.eta[mask])),
            looseB = squash(jets.passLooseB[mask]),
            mediumB = squash(jets.passMediumB[mask]),
            tightB = squash(jets.passTightB[mask]),
            flavor = squash(jets.hadronFlavour[mask]),
            weight = squash(wt_b[mask])
        )

        return {
            "Beff" : H
        }
