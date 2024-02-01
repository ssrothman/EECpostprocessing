import pickle
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from plotting.plotMatch import *

folder_nocut = 'Jan30_2024_nocuts_nothresh'
folder_cut = 'Jan31_2024_pythia_highstats_fixed_fixed'

with open("%s/matching/hists.pkl"%folder_nocut, 'rb') as f:
    hists_nocut = pickle.load(f)

with open("%s/matching/hists.pkl"%folder_cut, 'rb') as f:
    hists_cut = pickle.load(f)

pdgid = 211
match = 'matchReco'
etabin = None

plotMatchRate(hists_nocut['TrackDRMatch'][match], var = 'pt',
              label = 'No cuts', show=False, 
              etabin=etabin, pdgid = pdgid)

plotMatchRate(hists_cut['TrackDRMatch'][match], var = 'pt',
              label = 'PV cuts', show=False, 
              etabin=etabin, pdgid = pdgid)

plt.savefig("plots_smphadfeb2/vtxcuts_reco_matching.png")
plt.show()

match = 'matchGen'

plotMatchRate(hists_nocut['TrackDRMatch'][match], var = 'pt',
              label = 'No cuts', show=False, 
              etabin=etabin, pdgid = pdgid)

plotMatchRate(hists_cut['TrackDRMatch'][match], var = 'pt',
              label = 'PV cuts', show=False, 
              etabin=etabin, pdgid = pdgid)

plt.savefig("plots_smphadfeb2/vtxcuts_gen_matching.png")
plt.show()
