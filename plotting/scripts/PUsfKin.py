from plotting.util import *
from plotting.plotKin import *
from samples.latest import SAMPLE_LIST
import matplotlib.pyplot as plt

xsecs_D = config.xsecs

noSF = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['noPrefireSF',
                                                         'noIDsfs',
                                                         'noIsosfs',
                                                         'noTriggersfs',
                                                         'tight'])

nom = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight'])

PUup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                         'PUUP'])
PUdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                         'PUDN'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", ['tight'])
lumi = config.totalLumi

sigxsec = xsecs_D.DYJetsToLL
print(sigxsec)

PUupdn = [nom,
          PUup, PUdn]
PUupdnlabels = [
        'Nominal',
        'PU up', 'PU down'
        ]

folder = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'PU')

plotAllKin(data, lumi,
           PUupdn, 
           [sigxsec,sigxsec]*100,
           PUupdnlabels,
           btag=None,
           normToLumi=True,
           stack=False,
           folder = folder,
           show=False,
           done=True,
           density=False)

