import unfolding.detectormodel
import unfolding.histogram

from unfolding.detectormodel import load_hist_from_dataset
from unfolding.specs import dsspec

pythia_inclusive : dsspec = {
    'location' : 'local-submit',
    'dataset' : 'Pythia_inclusive',
    'runtag' : 'Apr_23_2025',
    'config_suite' : 'BasicConfig',
    'isMC' : True
}

gen = unfolding.histogram.Histogram.from_disk(
    'test_unfold/gen'
)
reco = unfolding.histogram.Histogram.from_disk(
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

model = unfolding.detectormodel.DetectorModel.from_disk(
    'test_unfold/detectormodel'
)