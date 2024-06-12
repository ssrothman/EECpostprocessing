from plotting.util import *
from plotting.plotKin import *
from samples.latest import SAMPLE_LIST
import matplotlib.pyplot as plt

xsecs_D = config.xsecs

nom = SAMPLE_LIST.get_hist('DYJetsToLL_allHT', "Kin", ['tight'])
Zreweight = SAMPLE_LIST.get_hist('DYJetsToLL_allHT', "Kin", ['tight',
                                                             'Zreweight'])

backgrounds = ['ZZ', 'WZ', 'WW', 
               'TTTo2L2Nu',
               'ST_tW_antitop', 'ST_tW_top',
               'ST_t_antitop_5f', 'ST_t_top_5f']

signals = [nom, Zreweight]
signal_labels = ['Inclusive', 'Z reweight']
signal_xsecs = [xsecs_D.DYJetsToLL, xsecs_D.DYJetsToLL]

background_hists = []
background_xsecs = []
for background in backgrounds:
    H = SAMPLE_LIST.get_hist(background, "Kin", ['tight'])
    if background.endswith("5f"):
        xsec = vars(xsecs_D)[background[:-3]]
    else:
        xsec = vars(xsecs_D)[background]

    background_hists.append(H)
    background_xsecs.append(xsec)

data = SAMPLE_LIST.get_hist('DATA_2018UL', "Kin", ['tight'])
lumi = config.totalLumi

plotAllKin(data, lumi,
           background_hists, background_xsecs, backgrounds,
           signals, signal_xsecs, signal_labels,
           btag=None,
           normToLumi=True,
           folder = None,
           show=True,
           done=True,
           density=False)
