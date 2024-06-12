from plotting.util import *
from plotting.plotBeff import *
from samples.latest import SAMPLE_LIST

import matplotlib.pyplot as plt

H = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Beff", ['noBtagSF',
                                                             'genXsec'])

folder = os.path.join('plots', SAMPLE_LIST.tag, "Beff")

for flavor in [0, 1, 2]:
    for eta in [0, 1]:
        plotTaglevel(H, flavor, eta,
                     folder = folder)

for level in ['tight', 'medium', 'loose']:
    for flavor in [0, 1, 2]:
        plotBarrelEndcap(H, level, flavor,
                         folder = folder)
