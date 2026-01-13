from unfolding.detectormodel import DetectorModel
from unfolding.histogram import Histogram
from unfolding.loss import Loss
from unfolding.minimizer import Minimizer

from unfolding.detectormodel import load_hist_from_dataset
from unfolding.specs import dsspec

import numpy as np

pythia_inclusive : dsspec = {
    'location' : 'local-submit',
    'dataset' : 'Pythia_inclusive',
    'runtag' : 'Apr_23_2025',
    'config_suite' : 'BasicConfig',
    'isMC' : True
}

gen = Histogram.from_disk(
    'test_unfold/gen'
)
reco = Histogram.from_disk(
    'test_unfold/reco'
)

unmatchedGen = load_hist_from_dataset(
    pythia_inclusive,
    'nominal',
    'res4tee_unmatchedGen_BINNED_nominal.npy'
)
untransferedGen = load_hist_from_dataset(
    pythia_inclusive,
    'nominal',
    'res4tee_untransferedGen_BINNED_nominal.npy'
)
bkgGen = unmatchedGen + untransferedGen

unmatchedReco = load_hist_from_dataset(
    pythia_inclusive,
    'nominal',
    'res4tee_unmatchedReco_BINNED_nominal.npy'
)
untransferedReco = load_hist_from_dataset(
    pythia_inclusive,
    'nominal',
    'res4tee_untransferedReco_BINNED_nominal.npy'
)
bkgReco = unmatchedReco + untransferedReco

model = DetectorModel.from_disk(
    'test_unfold/detectormodel'
)

baseline = gen.values[:]
baseline *= np.random.uniform(0.8, 1.2, size=baseline.shape)

loss = Loss(
    reco,
    genbaseline = baseline,
    model = model,
    negativePenalty = 1e6
)

mincfg = {
    'logpath' : 'test_unfold/minimization',
    'method' : 'l-bfgs',
    'cpt_interval' : 10,
    'cpt_start' : 0,
    'method_options' : {
    }
}


minimizer = Minimizer(mincfg)

minimizer(
    loss,
    x0 = None,
    device = 'cpu',
)