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

JERup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'JERUP'])
JERdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'JERDN'])
JESup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'JESUP'])
JESdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'JESDN'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", ['tight'])
lumi = config.totalLumi

sigxsec = xsecs_D.DYJetsToLL
print(sigxsec)

jetupdn = [nom,
           JERup, JERdn,
           JESup, JESdn]
jetupdnlabels = [
        'Nominal',
        'JER up', 'JER down',
        'JES up', 'JES down'
        ]

folder = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'JERJES')

plotAllKin(data, lumi,
           jetupdn, 
           [sigxsec,sigxsec]*100,
           jetupdnlabels,
           btag=None,
           normToLumi=True,
           stack=False,
           folder = folder,
           show=False,
           done=True,
           density=False)

