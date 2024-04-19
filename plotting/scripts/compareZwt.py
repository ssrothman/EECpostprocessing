from plotting.plotKin import plotKin, plotAllKin
from samples.latest import SAMPLE_LIST
from plotting.util import config
import os

nom = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", None)
Zwt = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['Zreweight'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", None)

plotAllKin(data, config.totalLumi, 
           [nom, Zwt], [config.xsecs.DYJetsToLL]*2,
           ['Nominal', "Zreweight"],
           normToLumi=True,
           stack=False,
           folder=None,
           show=True,
           done=True,
           density=False)
