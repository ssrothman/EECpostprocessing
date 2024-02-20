import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import json
from importlib import reload
import plotting.EECutil
import plotting.plotEEC

CORR = plotting.EECutil.EEChistReader('Feb19_2024_CORR/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/EEC')
SUM = plotting.EECutil.EEChistReader('Feb19_2024_SUM/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/EEC')
RAW = plotting.EECutil.EEChistReader('Feb19_2024_RAW/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/EEC')

bins = {'order' : 0, 'pt' : 2}
