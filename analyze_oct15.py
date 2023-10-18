import pickle

AOD = {}
miniAOD = {}

names = ['dRscan1', 'dRscan2', 'dRscan3']

import itertools

for name in names:
    print(name)
    with open("/home/submit/srothman/data/EEC/Oct15_2023_dRscan_AOD_take3/%s/hists.pkl"%(name), 'rb') as f:
        AOD.update(pickle.load(f))

#with open("/home/submit/srothman/data/EEC/Oct10_2023_filterscan_fixedcuts_AOD/jets4/hists.pkl", 'rb') as f:
#    AOD.update(pickle.load(f))
