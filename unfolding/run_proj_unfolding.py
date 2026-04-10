import fasteigenpy  # must import before torch

import os
import numpy as np

from unfolding.detectormodel import DetectorModel
from unfolding.histogram import Histogram
from unfolding.loss import Loss
from unfolding.minimizer import Minimizer

WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace'

# --- load workspace ---
print("Loading reco histogram...")
reco = Histogram.from_disk(os.path.join(WORKSPACE, 'reco'))

print("Loading gen histogram...")
gen = Histogram.from_disk(os.path.join(WORKSPACE, 'gen'))

print("Loading detector model...")
model = DetectorModel.from_disk(os.path.join(WORKSPACE, 'detectormodel'))
print(model)
valid = np.load(os.path.join(WORKSPACE, 'valid_bins.npy'))
ntrim = (model.nReco - len(valid)) // 2 
model._transfer0          = model._transfer0[ntrim:-ntrim, ntrim:-ntrim]
model._gamma0             = model._gamma0[ntrim:-ntrim]
model._rho0               = model._rho0[ntrim:-ntrim]
model._gammaVariations    = model._gammaVariations[:, ntrim:-ntrim]
model._rhoVariations      = model._rhoVariations[:, ntrim:-ntrim]
model._transferVariations = model._transferVariations[:, ntrim:-ntrim, ntrim:-ntrim]
model._transfer0          = model._transfer0[np.ix_(valid, valid)]
model._gamma0             = model._gamma0[valid]
model._rho0               = model._rho0[valid]
model._gammaVariations    = model._gammaVariations[:, valid]                                                                          
model._rhoVariations      = model._rhoVariations[:, valid]                                                                            
model._transferVariations = model._transferVariations[:, np.ix_(valid, valid)[0], np.ix_(valid, valid)[1]]
model._nGen  = int(valid.sum())                                                                                                       
model._nReco = int(valid.sum())
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
