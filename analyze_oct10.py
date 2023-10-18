import pickle

AOD = {}
miniAOD = {}

do = 'HAD0'

if do == 'ELE':
    for name in ['ELE']:
        with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))

    for name in ["ELERecover", "ELEloose", "ELElooseRecover", 'ELElooseRecoverDrop']:
        with open("/home/submit/srothman/data/EEC/Oct09_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))
elif do == 'MU':
    for name in ['MU']:
        with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))

    for name in ["MURecover", "MUloose", "MUlooseRecover", 'MUlooseRecoverDrop']:
        with open("/home/submit/srothman/data/EEC/Oct09_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))
elif do == 'HADCH':
    for name in ['HADCH']:
        with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))

    for name in ["HADCHRecover", "HADCHloose", "HADCHlooseRecover", 'HADCHlooseRecoverDrop']:
        with open("/home/submit/srothman/data/EEC/Oct09_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))
elif do == 'EM0':
    for name in ['EM0','EM0Drop']:
        with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))

    for name in ["EM0Recover", "EM0looseRecover", 'EM0looseRecoverDrop']:
        with open("/home/submit/srothman/data/EEC/Oct09_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))
elif do == 'HAD0':
    for name in ['HAD0','HAD0Drop']:
        with open("/home/submit/srothman/data/EEC/Oct08_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))

    for name in ["HAD0Recover", "HAD0looseRecover", 'HAD0looseRecoverDrop']:
        with open("/home/submit/srothman/data/EEC/Oct09_2023_cutscan_chargedscan/%s/hists.pkl"%name, 'rb') as f:
            miniAOD.update(pickle.load(f))
