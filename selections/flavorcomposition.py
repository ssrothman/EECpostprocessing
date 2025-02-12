import awkward as ak
import numpy as np

def getSpecificFlavorWeights(recojets, config, whichflav):
    flavmask = ak.any(recojets.jets.hadronFlavour == whichflav, axis=-1)
    
    factor = vars(config.flavorSystematicFactors)[str(whichflav)]

    w_nom = np.ones(len(flavmask))
    w_up = np.ones(len(flavmask))
    w_down = np.ones(len(flavmask))

    w_up[flavmask] = 1+factor
    w_down[flavmask] = 1-factor

    return w_nom, w_up, w_down

def getAllFlavorWeights(weights, recojets, config):
    flavs = [0, 4, 5, 21]
    names = ['uds', 'c', 'b', 'g']

    for flav, name in zip(flavs, names):
        w_nom, w_up, w_down = getSpecificFlavorWeights(recojets, config, flav)

        weights.add("wt_%s_xsec"%name, w_nom, w_up, w_down)
