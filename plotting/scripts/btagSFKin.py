from plotting.util import *
from plotting.plotKin import *
from samples.latest import SAMPLE_LIST
import matplotlib.pyplot as plt

xsecs_D = config.xsecs

nom = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight'])
btagup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'btagSFUP'])
btagdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                           'btagSFDN'])
PDFup = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'PDFaSUP'])
PDFdn = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight',
                                                          'PDFaSDN'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", ['tight'])
lumi = config.totalLumi

sigxsec = xsecs_D.DYJetsToLL
print(sigxsec)

hists = [nom,
            btagup, btagdn,
            PDFup, PDFdn]
labels = [
        'Nominal',
        'btag up', 'btag down',
        'PDF up', 'PDF down'
        ]

folder = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagSF')
folder_inc = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagSF', 'inclusive')
folder_pass = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagSF', 'pass')
folder_fail = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagSF', 'fail')

plotAllKin(data, lumi,
           hists, 
           [sigxsec,sigxsec]*100,
           labels,
           btag=None,
           normToLumi=True,
           stack=False,
           folder = folder_inc,
           show=False,
           done=True,
           density=False)

plotAllKin(data, lumi,
           hists, 
           [sigxsec,sigxsec]*100,
           labels,
           btag=0,
           normToLumi=True,
           stack=False,
           folder = folder_fail,
           show=False,
           done=True,
           density=False)

plotAllKin(data, lumi,
           hists, 
           [sigxsec,sigxsec]*100,
           labels,
           btag=1,
           normToLumi=True,
           stack=False,
           folder = folder_pass,
           show=False,
           done=True,
           density=False)

