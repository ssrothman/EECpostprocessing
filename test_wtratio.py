import pickle

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/gen_nominal_nominal.pkl", 'rb') as f:
    Hgen = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/unmatchedGen_nominal_nominal.pkl", 'rb') as f:
    HunmatchedGen = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/untransferedGen_nominal_nominal.pkl", 'rb') as f:
    HuntransferedGen = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/reco_nominal_nominal.pkl", 'rb') as f:
    Hreco = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/unmatchedReco_nominal_nominal.pkl", 'rb') as f:
    HunmatchedReco = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/untransferedReco_nominal_nominal.pkl", 'rb') as f:
    HuntransferedReco = pickle.load(f)[{'bootstrap' : 0}]

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/transfer_nominal_nominal.pkl", 'rb') as f:
    Htransfer = pickle.load(f)

with open("/ceph/submit/data/user/s/srothman/EEC/Apr_01_2025/Pythia_inclusive/EECres4tee/hists_file0to208_MC/nominal/wtratio_nominal_nominal.pkl", 'rb') as f:
    Hwtratio, H_wtgen, H_wtreco = pickle.load(f)

HgenPure = Hgen - HunmatchedGen - HuntransferedGen
HrecoPure = Hreco - HunmatchedReco - HuntransferedReco

genpure = HgenPure.values(flow=True)
recopure = HrecoPure.values(flow=True)

transfer = Htransfer.values(flow=True)
transfer = transfer/genpure[None,None,None,None,:,:,:,:]
transfersum = transfer.sum(axis=(0,1,2,3))
