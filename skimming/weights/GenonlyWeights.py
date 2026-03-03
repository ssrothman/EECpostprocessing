from skimming.objects.AllObjects import AllObjects
from coffea.analysis_tools import Weights

class GenonlyWeights:
    def __init__(self, cfg : dict, evtselcfg : dict):
        self._cfg = cfg
        self._evtselcfg = evtselcfg

    def get_weights(self, allobjects : AllObjects) -> Weights:

        wts = Weights(len(allobjects.event), storeIndividual=True)

        if not allobjects.isMC:
            return wts #short circuit for data

        wts.add('generator', allobjects.genWeight)

        return wts