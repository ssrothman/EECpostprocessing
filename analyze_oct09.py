import pickle

AOD = {}
miniAOD = {}

for name in ['HADCH']:
    with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan_AOD/%s/hists.pkl"%name, 'rb') as f:
        AOD.update(pickle.load(f))

    #with open("/home/submit/srothman/work/EEC/postprocessing/Oct08_2023_cutscan_chargedscan_AOD/%s/hists.pkl"%name, 'rb') as f:
    #    AOD.update(pickle.load(f))

    with open("/home/submit/srothman/work/EEC/postprocessingOct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
        miniAOD.update(pickle.load(f))
