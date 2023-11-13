import pickle

#with open("Nov01_2023_EECcorr_caloreco_fixed_fixed_herwig/EEC/hists.pkl", 'rb') as f:
#    hists = pickle.load(f)

hists = {}

with open("Nov01_2023_highstats/EEC/hists.pkl", 'rb') as f:
    pythia = pickle.load(f)

for key in pythia:
    hists['pythia_'+key] = pythia[key]

with open("Nov01_2023_highstats_herwig/EEC/hists.pkl", 'rb') as f:
    herwig = pickle.load(f)

for key in herwig:
    hists['herwig_'+key] = herwig[key]
