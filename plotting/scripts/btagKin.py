from plotting.util import *
from plotting.plotKin import *
from samples.latest import SAMPLE_LIST
import matplotlib.pyplot as plt

xsecs_D = config.xsecs

noSF = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['noBtagSF'])
tight = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", ['tight'])

ZZ = SAMPLE_LIST.lookup("ZZ").get_hist("Kin", ['tight'])
WZ = SAMPLE_LIST.lookup("WZ").get_hist("Kin", ['tight'])
WW = SAMPLE_LIST.lookup("WW").get_hist("Kin", ['tight'])
TT = SAMPLE_LIST.lookup("TTTo2L2Nu").get_hist("Kin", ['tight'])

ST_tW_antitop = SAMPLE_LIST.lookup("ST_tW_antitop").get_hist("Kin", ['tight'])
ST_tW_top = SAMPLE_LIST.lookup("ST_tW_top").get_hist("Kin", ['tight'])

ST_t_antitop = SAMPLE_LIST.lookup("ST_t_antitop_5f").get_hist("Kin", ['tight'])
ST_t_top = SAMPLE_LIST.lookup("ST_t_top_5f").get_hist("Kin", ['tight'])

data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", None)
lumi = config.totalLumi

bkgs = ['ST_t_antitop', 'ST_t_top',
        'ST_tW_antitop', 'ST_t_top',
        'WW', 'ZZ', 'WZ',
        'TTTo2L2Nu']
bkgxsecs = [vars(xsecs_D)[bkg] for bkg in bkgs]
bkghists = [
    ST_t_antitop, ST_t_top,
    ST_tW_antitop, ST_tW_top,
    WW, ZZ, WZ,
    TT
]


sigs = [noSF, tight]
labels =  ['noBtagSF', 'tight']
sigxsec = xsecs_D.DYJetsToLL

#plotKin(data, lumi,
#        [noSF, tight],
#        [sigxsec, sigxsec],
#        ['noBtagSF', 'tight'],
#        'HJet', 'Jpt',
#        logx=True, logy=True,
#        btag=1,
#        normToLumi=True,
#        stack=False,
#        folder = None,
#        show=True,
#        done=True,
#        density=False)
#
#plotKin(data, lumi,
#        [noSF, tight],
#        [sigxsec, sigxsec],
#        ['noBtagSF', 'tight'],
#        'HJet', 'Jpt',
#        logx=True, logy=True,
#        btag=0,
#        normToLumi=True,
#        stack=False,
#        folder = None,
#        show=True,
#        done=True,
#        density=False)
#
#plotKin(data, lumi,
#        [noSF, tight],
#        [sigxsec, sigxsec],
#        ['noBtagSF', 'tight'],
#        'HJet', 'Jpt',
#        logx=True, logy=True,
#        btag=None,
#        normToLumi=True,
#        stack=False,
#        folder = None,
#        show=True,
#        done=True,
#        density=False)

folder_pass = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagPass')
folder_fail = os.path.join('plots', SAMPLE_LIST.tag, 'Kin', 'btagFail')

for sig, label in zip(sigs, labels):
    plotKin(data, lumi, 
            bkghists+[sig], 
            bkgxsecs+[sigxsec],
            bkgs+[label],
            'HJet', 'Jpt',
            logx=True, logy=True,
            btag=1,
            normToLumi=True,
            stack=True,
            folder = folder_pass,
            show=True,
            done=True,
            density=False)

    plotKin(data, lumi, 
            bkghists+[sig], 
            bkgxsecs+[sigxsec],
            bkgs+[label],
            'HJet', 'Jeta',
            logx=False, logy=True,
            btag=1,
            normToLumi=True,
            stack=True,
            folder = folder_pass,
            show=True,
            done=True,
            density=False)

    plotKin(data, lumi, 
            bkghists+[sig], 
            bkgxsecs+[sigxsec],
            bkgs+[label],
            'HJet', 'Jpt',
            logx=True, logy=True,
            btag=0,
            normToLumi=True,
            stack=True,
            folder = folder_fail,
            show=True,
            done=True,
            density=False)

    plotKin(data, lumi, 
            bkghists+[sig], 
            bkgxsecs+[sigxsec],
            bkgs+[label],
            'HJet', 'Jeta',
            logx=False, logy=True,
            btag=0,
            normToLumi=True,
            stack=True,
            folder = folder_fail,
            show=True,
            done=True,
            density=False)
