from math import e
from skimming.objects.AllObjects import AllObjects
from skimming.weights.GenonlyWeights import GenonlyWeights
from .StandardWeights import StandardWeights
from coffea.analysis_tools import Weights

weightsclasses = {
    'StandardWeights' : StandardWeights,
    "GenonlyWeights" : GenonlyWeights 
}

def runWeightsFactory(wtcfg : dict,
                      evtselcfg : dict,
                      allobjects : AllObjects) -> Weights:

    weightsclass = weightsclasses[wtcfg['class']]
    obj = weightsclass(wtcfg['params'], evtselcfg['params'])
    weights = obj.get_weights(allobjects)
    return weights