import pickle
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

folderP = 'Jan25_2024_puppiCut/matching'
folderZ = 'Jan25_2024_dzcut/matching'
folderZXY = 'Jan25_2024_dzdxycut/matching'

with open("%s/hists.pkl"%folderP, 'rb') as f:
    histsP = pickle.load(f)

with open("%s/hists.pkl"%folderZ, 'rb') as f:
    histsZ = pickle.load(f)

with open("%s/hists.pkl"%folderZXY, 'rb') as f:
    histsZXY = pickle.load(f)

from plotting.plotMatch import *

pdgid = 130
match = 'matchReco'

plotMatchRate(histsP['TrackDRMatch'][match], var = 'pt',
              label = 'puppi weight cut', show=False, 
              etabin=0, pdgid = pdgid)

plotMatchRate(histsZ['TrackDRMatch'][match], var = 'pt',
              label = 'dz cut', show=False, 
              etabin=0, pdgid = pdgid)

plotMatchRate(histsZXY['TrackDRMatch'][match], var = 'pt',
              label = 'dz,dxy cut', show=False, 
              etabin=0, pdgid = pdgid)

plt.legend()
plt.show()
