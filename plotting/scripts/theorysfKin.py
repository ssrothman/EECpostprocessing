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

PDFup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'PDFaSUP'])
PDFdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'PDFaSDN'])
FSRup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'FSRUP'])
FSRdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'FSRDN'])
ISRup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'ISRUP'])
ISRdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'ISRDN'])
scaleup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                            'scaleUP'])
scaledn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                            'scaleDN'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", ['tight'])
lumi = config.totalLumi

sigxsec = xsecs_D.DYJetsToLL
print(sigxsec)

theoryupdn = [nom,
              PDFup, PDFdn,
              FSRup, FSRdn,
              ISRup, ISRdn,
              scaleup, scaledn]
theoryupdnlabels = [
        'Nominal',
        'PDF up', 'PDF down',
        'FSR up', 'FSR down',
        'ISR up', 'ISR down',
        'Scale up', 'Scale down'
        ]

folder = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'theorySFs')

plotAllKin(data, lumi,
           theoryupdn, 
           [sigxsec,sigxsec]*100,
           theoryupdnlabels,
           btag=None,
           normToLumi=True,
           stack=False,
           folder = folder,
           show=False,
           done=True,
           density=False)

