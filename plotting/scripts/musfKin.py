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

IDup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                         'idsfUP'])
IDdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                         'idsfDN'])
ISOup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'isosfUP'])
ISOdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'isosfDN'])
trigUP = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'triggersfUP'])
trigDN = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'triggersfDN'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", ['tight'])
lumi = config.totalLumi

sigxsec = xsecs_D.DYJetsToLL

MUupdn = [nom,
          IDup, IDdn,
          ISOup, ISOdn,
          trigUP, trigDN]
MUupdnlabels = [
        'Nominal',
        'ID up', 'ID down',
        'ISO up', 'ISO down',
        'Trig up', 'Trig down'
        ]

folder = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'muSFs')

plotAllKin(data, lumi,
           MUupdn,
           [sigxsec,sigxsec]*100,
           MUupdnlabels,
           btag=None,
           normToLumi=True,
           stack=False,
           folder = folder,
           show=False,
           done=True,
           density=False)

