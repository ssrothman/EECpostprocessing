import pickle

AOD = {}
miniAOD = {}

names = ['Charged_2', 'CaloShare', 'Recover', 'ChargedSupplement', 'ExtraHists']

import itertools

for name in names:
    print(name)
    with open("/home/submit/srothman/data/EEC/Oct14_2023_filterscan_AOD/%s/hists.pkl"%(name), 'rb') as f:
        AOD.update(pickle.load(f))

#with open("/home/submit/srothman/data/EEC/Oct10_2023_filterscan_fixedcuts_AOD/jets4/hists.pkl", 'rb') as f:
#    AOD.update(pickle.load(f))
