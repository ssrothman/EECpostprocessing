import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from correctionlib import CorrectionSet

from .theorySF import getAllTheorySFs
from .muonSF import getAllMuonSFs
from .btagSF import getBtagSF
from .reweightingSF import getZptSF, getPUweight


def getEventWeight(x, readers, config, isMC,
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

    getAllTheorySFs(ans, readers)
    getAllMuonSFs(ans,
                  readers,
                  config,
                  noPrefireSF,
                  noIDsfs,
                  noIsosfs,
                  noTriggersfs)

    if Zreweight:
        getZptSF(ans, raders.Zs, config)
        
    if not noPUweight:
        getPUweight(ans, readers.nTrueInt, config, isMC)

    if not noBtagSF:
        getBtagSF(ans, readers.rRecoJet, config)

    return ans
