import pickle

AOD = {}
miniAOD = {}

for elecut in ['Tight', 'Had', 'Loose', 'LooseNoMu']:
    for phocut in ['Tight', 'Loose']:
        with open("/home/submit/srothman/data/EEC/Oct10_2023_filterscan_AOD/%sEle%sPho/hists.pkl"%(elecut, phocut), 'rb') as f:
            AOD.update(pickle.load(f))

#    with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
#        miniAOD.update(pickle.load(f))
