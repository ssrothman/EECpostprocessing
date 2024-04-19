from plotting.plotKin import plotKin, plotAllKin
from samples.latest import SAMPLE_LIST
from plotting.util import config
import os

nom = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Kin", None)
xsecdb1 = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Kin", ["xsecDB"])
xsecdb2 = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Kin", ["xsecDB2"])
inclusive = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", None)

plotAllKin(inclusive, [1], [nom, xsecdb1, xsecdb2], 
           [1, 1, 1], 
           ["Tuned", "xsecDB", "xsecDB2"], 
           normToLumi=False, 
           stack=False,
           folder=None,
           show=True, 
           done=True, 
           density=True,
           dataname='Inclusive Sample')
