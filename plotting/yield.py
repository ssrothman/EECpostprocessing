from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import simonplot as splt
import numpy as np

signal = build_pq_dataset_stack(
    'BasicConfig',
    'Mar_01_2026',
    'Pythia_HTsum',
    'nominal',
    'jets',
    location = 'scratch-submit',
)
bkg = build_pq_dataset_stack(
    'BasicConfig',
    'Mar_01_2026',
    'allBKG',
    'nominal',
    'jets',
    location = 'scratch-submit',
)

lumi = 59.81

bkg.compute_weight(lumi)
signal.compute_weight(lumi)

basecut = splt.cut.AndCuts([
    splt.cut.EqualsCut('nMu', 2),
    splt.cut.EqualsCut('nEle', 0),
    splt.cut.LessThanCut('numTightB', 2),
    splt.cut.TwoSidedCut('Zmass', 91.1876-10, 91.1876+10)
])

basecut = splt.cut.NoCut()

ptbins = [
    splt.cut.TwoSidedCut('Jpt', 30, 50),
    splt.cut.TwoSidedCut('Jpt', 50, 90),
    splt.cut.TwoSidedCut('Jpt', 90, 150),
    splt.cut.TwoSidedCut('Jpt', 150, 250),
    splt.cut.TwoSidedCut('Jpt', 250, 400),
    splt.cut.TwoSidedCut('Jpt', 400, np.inf)
]

for ptbin in ptbins:
    print("Yields for pT bin", ptbin.key)
    bkgyield = bkg.estimate_yield(splt.cut.AndCuts([basecut, ptbin]), splt.variable.BasicVariable('wt_nominal'))
    signalyield = signal.estimate_yield(splt.cut.AndCuts([basecut, ptbin]), splt.variable.BasicVariable('wt_nominal'))
    totalyield = bkgyield + signalyield

    print("\tBackground: %0.0f (%0.4g%%)"%(bkgyield, 100*bkgyield/totalyield))
    print("\tSignal: %0.0f (%0.4g%%)"%(signalyield, 100*signalyield/totalyield))
    print("\tTotal: %0.0f"%(totalyield))