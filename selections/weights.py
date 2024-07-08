import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

from .theorySF import getAllTheorySFs
from .muonSF import getAllMuonSFs
from .btagSF import getBtagSF
from .reweightingSF import getZptSF, getPUweight


def getEventWeight(x, muons, Zs, rRecoJet, config, isMC,
                   noPUweight,
                   noPrefireSF,
                   noIDsfs,
                   noIsosfs,
                   noTriggersfs,
                   noBtagSF,
                   Zreweight):
    ans = Weights(len(x), storeIndividual=True)

    if not isMC:
        return

    getAllTheorySFs(ans, x)
    getAllMuonSFs(ans,
                  x.L1PreFiringWeight,
                  muons,
                  config,
                  noPrefireSF,
                  noIDsfs,
                  noIsosfs,
                  noTriggersfs)

    if Zreweight:
        getZptSF(ans, Zs, config)
        
    if not noPUweight:
        getPUweight(ans, x.Pileup.nTrueInt, config, isMC)

    if not noBtagSF:
        getBtagSF(ans, rRecoJet, config)

    return ans
