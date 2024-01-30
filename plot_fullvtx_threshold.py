import pickle
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

folder = 'Jan26_2024_puppidzdxycut_thresholds/matching'

with open("%s/hists.pkl"%folder, 'rb') as f:
    hists = pickle.load(f)

from plotting.plotMatch import *

pdgids = [11, 13, 211, 22, 130]
names = ['Ele', 'Mu', "HADCH", 'EM0', 'HAD0']
match = 'matchReco'
etabin = 0

for pdgid, name in zip(pdgids, names):
    plotMatchRate(hists['TrackDRMatch'][match], var = 'pt',
                  label = name, show=False, 
                  etabin=etabin, pdgid = pdgid)

plt.legend()
plt.show()
