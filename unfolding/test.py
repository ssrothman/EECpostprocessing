import fasteigenpy # need to import first
                   # so that later torch imports
                   # don't break fasteigenpy

from unfolding.specs import detectormodelspec, dsspec
from unfolding.unfolding_workspace import setup_unfolding_workspace

pythia_inclusive : dsspec = {
    'location' : 'local-submit',
    'dataset' : 'Pythia_inclusive',
    'runtag' : 'Apr_23_2025',
    'config_suite' : 'BasicConfig',
    'isMC' : True
}

basicmodel : detectormodelspec = {
    'systematics' : []
}

setup_unfolding_workspace(
    'test_unfold',
    pythia_inclusive,
    'nominal',
    'nominal',
    pythia_inclusive,
    basicmodel,
    'res4tee'
)