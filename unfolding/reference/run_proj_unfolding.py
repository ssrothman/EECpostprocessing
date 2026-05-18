import fasteigenpy  # must import before torch

import os
import numpy as np

from unfolding.detectormodel import DetectorModel
from unfolding.histogram import Histogram
from unfolding.loss import Loss
from unfolding.minimizer import Minimizer

WORKSPACE = '/eos/user/d/dponman/proj_unfold_workspace_data'

# --- load workspace ---
print("Loading reco histogram...")
reco = Histogram.from_disk(os.path.join(WORKSPACE, 'reco'))

print("Loading gen histogram...")
gen = Histogram.from_disk(os.path.join(WORKSPACE, 'gen'))

print("Loading detector model...")
model = DetectorModel.from_disk(os.path.join(WORKSPACE, 'detectormodel'))
print(model)
genbaseline = gen.values.copy()
print("reco invcov diagonal [:10]:", reco.invcov.diagonal()[:10])
print("reco invcov norm:", np.linalg.norm(reco.invcov))
print("reco cov diagonal [:10]:", reco.covmat.diagonal()[:10])

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

pred = model.forward(gen.values.copy(), np.zeros(model.nSyst))
print("self-consistency check:")
print("  gen sum:  ", gen.values.sum())
print("  pred sum: ", pred.sum())
print("  reco sum: ", reco.values.sum())
print("  max rel diff:", np.max(np.abs(pred - reco.values) /
np.maximum(reco.values, 1e-10)))
print("  gamma0 mean:    ", model._gamma0.mean())
print("  rho0 mean:      ", model._rho0.mean())
print("  t0 row sums [:5]:", model._transfer0.sum(axis=1)[:5])
print("  (1-gamma) * gen sum:", (gen.values * (1 - model._gamma0)).sum())
minimizer = Minimizer(mincfg)
minimizer(loss, x0=None, device='cpu')
