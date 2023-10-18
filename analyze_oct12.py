import pickle

AOD = {}
miniAOD = {}

names = ['NeutralFlavorFilters', 'ChargedFilters', 'RecoverPt1', 'RecoverPt2', 'Thresholds', 'Drop']

import itertools

for name in names:
    with open("/home/submit/srothman/data/EEC/Oct12_2023_filterscan_AOD_take2/%s/hists.pkl"%(name), 'rb') as f:
        AOD.update(pickle.load(f))

#with open("/home/submit/srothman/data/EEC/Oct10_2023_filterscan_fixedcuts_AOD/jets4/hists.pkl", 'rb') as f:
#    AOD.update(pickle.load(f))
