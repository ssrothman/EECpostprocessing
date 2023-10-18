import pickle

with open("Oct06_2023_cutoffscan_fixed_highstats/EM0/hists.pkl", 'rb') as f:
    EM0 = pickle.load(f)

with open("Oct06_2023_cutoffscan_fixed_highstats/HAD0/hists.pkl", 'rb') as f:
    HAD0 = pickle.load(f)

with open("Oct06_2023_cutoffscan_fixed_highstats/HADCH/hists.pkl", 'rb') as f:
    HADCH = pickle.load(f)

with open("Oct06_2023_cutoffscan_fixed_highstats/ELE/hists.pkl", 'rb') as f:
    ELE = pickle.load(f)

with open("Oct06_2023_cutoffscan_fixed_highstats/MU/hists.pkl", 'rb') as f:
    MU = pickle.load(f)

import plotting.plotMatch
import post.resolution

cuts = [1,2,3,4,5,6,7,10,15,20]
hists = [EM0, HAD0, HADCH, ELE, MU]
names = ['EM0', 'HAD0', 'HADCH', 'ELE', 'MU']
baseDRs = [0.01, 0.01, 0.001, 0.001, 0.001]

folder = 'Oct06_2023_cutoffscan_fixed_highstats'

for hist, name, baseDR in zip(hists, names, baseDRs):
    for etabin in [0, 1, 2]:
        plotting.plotMatch.cutscan(hist, cuts, name, 
                                   etabin, baseDR, folder)
        plotting.plotMatch.cutscanefficiency(hist, cuts, name, 
                                             etabin, baseDR, 10, folder)
        plotting.plotMatch.cutscanefficiency(hist, cuts, name, 
                                             etabin, baseDR, 30, folder)
        plotting.plotMatch.cutscanefficiency(hist, cuts, name, 
                                             etabin, baseDR, 50, folder)
        post.resolution.cutscan(hist, cuts, name, etabin, baseDR, folder)

