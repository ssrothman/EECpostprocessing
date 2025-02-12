import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

def getZptSF(weights, Zs, config):
    cset = CorrectionSet.from_file(config.Zreweight.path)

    badpt = ak.is_none(Zs.pt)
    Zpt = ak.fill_none(Zs.pt, 0)

    bady = ak.is_none(Zs.rapidity) | (np.abs(Zs.rapidity) > 2.4)
    Zy = ak.fill_none(Zs.rapidity, 0)
    
    Zsf = cset['Zwt'].evaluate(Zpt, np.abs(Zy))

    Zsf = np.where(badpt | bady, 1, Zsf)
    Zsf = np.where(Zsf <=0, 1, Zsf) #protect against zeros.
                                    #shouldn't happen, but just in case

    weights.add('wt_Zkin', Zsf)

def getPUweight(weights, nTruePU, config, isMC):
    if not isMC:
        return

    cset = CorrectionSet.from_file(config.PUreweight.path)
    ev = cset[config.PUreweight.name]

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
