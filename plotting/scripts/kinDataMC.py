from plotting.plotKin import plotKin, plotAllKin
from samples.latest import SAMPLE_LIST
from plotting.util import config

def doKinDataMC(folder, show, tags=[]):
    xsecs_D = config.xsecs

    '''
    Compare amount of stats
    '''
    Pythia_inclusive = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", tags)
    Herwig_inclusive = SAMPLE_LIST.lookup("DYJetsToLL_Herwig").get_hist("Kin", tags)
    Pythia_HT = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Kin", tags)
    ZZ = SAMPLE_LIST.lookup("ZZ").get_hist("Kin", tags)
    WZ = SAMPLE_LIST.lookup("WZ").get_hist("Kin", tags)
    WW = SAMPLE_LIST.lookup("WW").get_hist("Kin", tags)
    TT = SAMPLE_LIST.lookup("TTTo2L2Nu").get_hist("Kin", tags)

    data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", tags)
    lumi = config.totalLumi

    bkgs = [WW, ZZ, WZ, TT]
    bkgxsecs = [xsecs_D.WW, xsecs_D.ZZ, xsecs_D.WZ, xsecs_D.TTTo2L2Nu]
    bkglabels = ['WW', 'ZZ', 'WZ', 'TTTo2L2Nu']

    sigs = [Pythia_inclusive, Pythia_HT, Herwig_inclusive]
    sigxsecs = [xsecs_D.DYJetsToLL]*3
    siglabels = ['Inclusive Pythia', 'HT-binned Pythia', 'Inclusive Herwig']

    for i in range(len(sigs)):
        plotAllKin(data, lumi, 
                   bkgs+[sigs[i]], 
                   bkgxsecs+[sigxsecs[i]], 
                   bkglabels+[siglabels[i]],
                normToLumi=True,
                stack=True,
                folder = folder,
                   show=show,
                   done=True,
                density=False)
