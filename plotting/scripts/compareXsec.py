from plotting.plotKin import plotKin, plotAllKin
from samples.latest import SAMPLE_LIST
from plotting.util import config
import os

HTnames = [
    'DYJetsToLL_HT-0to70',
    'DYJetsToLL_HT-70to100',
    'DYJetsToLL_HT-100to200',
    'DYJetsToLL_HT-200to400',
    'DYJetsToLL_HT-400to600',
    'DYJetsToLL_HT-600to800',
    'DYJetsToLL_HT-800to1200',
    'DYJetsToLL_HT-1200to2500',
    'DYJetsToLL_HT-2500toInf',
]
HThists = [SAMPLE_LIST.lookup(name).get_hist("Kin", ['noBtagSF']) for name in HTnames]
HTxsecs = [vars(config.xsecs)[name] for name in HTnames]

inclusive = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['noBtagSF'])

folder = os.path.join('plots', SAMPLE_LIST.tag, "xsecValidation")

plotAllKin(inclusive, 15.6, HThists, HTxsecs, HTnames, 
           normToLumi=True, 
           stack=True,
           density=False,
           dataname='HT-Inclusive',
           show=True, 
           folder=None,
           done=True)
