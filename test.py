import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import json
from importlib import reload
import plotting.EECutil
import plotting.plotEEC

x = plotting.EECutil.EEChistReader('Feb20_2024_charged_RAW/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/EEC')

bins = {'order' : 0, 'pt' : 2}
