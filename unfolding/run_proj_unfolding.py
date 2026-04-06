import fasteigenpy  # must import before torch

import os
import numpy as np

from unfolding.detectormodel import DetectorModel
from unfolding.histogram import Histogram
from unfolding.loss import Loss
from unfolding.minimizer import Minimizer

WORKSPACE = 'proj_unfold_workspace'

# --- load workspace ---
print("Loading reco histogram...")
reco = Histogram.from_disk(os.path.join(WORKSPACE, 'reco'))

print("Loading gen histogram...")
gen = Histogram.from_disk(os.path.join(WORKSPACE, 'gen'))

print("Loading detector model...")
model = DetectorModel.from_disk(os.path.join(WORKSPACE, 'detectormodel'))
print(model)

# gen values are used as the baseline; minimizer solves for per-bin multipliers
genbaseline = gen.values.copy()

loss = Loss(
    reco           = reco,
    genbaseline    = genbaseline,
    model          = model,
    negativePenalty = 1e6,
)

mincfg = {
    'logpath'        : '%s/minimization' % WORKSPACE,
    'method'         : 'l-bfgs',
    'cpt_interval'   : 10,
    'cpt_start'      : 0,
    'method_options' : {},
}

minimizer = Minimizer(mincfg)
minimizer(loss, x0=None, device='cpu')
