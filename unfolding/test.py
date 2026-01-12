from unfolding.unfolding_workspace import dsspec, setup_unfolding_workspace

pythia_inclusive : dsspec = {
    'location' : 'local-submit',
    'dataset' : 'Pythia_inclusive',
    'runtag' : 'Apr_23_2025',
    'config_suite' : 'BasicConfig',
    'isMC' : True
}

setup_unfolding_workspace(
    'test_unfold',
    pythia_inclusive,
    'nominal',
    'nominal',
    pythia_inclusive,
    'res4tee'
)