from plotting.plotKin import plotKin, plotAllKin
from samples.latest import SAMPLE_LIST
from plotting.util import config
import os

def doHTval(folder, show, tags=[]):
    xsecs_D = config.xsecs

    '''
    Compare amount of stats
    '''
    Pythia_inclusive = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", tags)
    Herwig_inclusive = SAMPLE_LIST.lookup("DYJetsToLL_Herwig").get_hist("Kin", tags)
    Pythia_0to70 = SAMPLE_LIST.lookup("DYJetsToLL_HT-0to70").get_hist("Kin", tags)
    Pythia_70to100 = SAMPLE_LIST.lookup("DYJetsToLL_HT-70to100").get_hist("Kin", tags)
    Pythia_100to200 = SAMPLE_LIST.lookup("DYJetsToLL_HT-100to200").get_hist("Kin", tags)
    Pythia_200to400 = SAMPLE_LIST.lookup("DYJetsToLL_HT-200to400").get_hist("Kin", tags)
    Pythia_400to600 = SAMPLE_LIST.lookup("DYJetsToLL_HT-400to600").get_hist("Kin", tags)
    Pythia_600to800 = SAMPLE_LIST.lookup("DYJetsToLL_HT-600to800").get_hist("Kin", tags)
    Pythia_800to1200 = SAMPLE_LIST.lookup("DYJetsToLL_HT-800to1200").get_hist("Kin", tags)
    Pythia_1200to2500 = SAMPLE_LIST.lookup("DYJetsToLL_HT-1200to2500").get_hist("Kin", tags)
    Pythia_2500toInf = SAMPLE_LIST.lookup("DYJetsToLL_HT-2500toInf").get_hist("Kin", tags)

    data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Kin", tags)

    baseline = None
    MCs = [Pythia_0to70, Pythia_70to100, Pythia_100to200, Pythia_200to400, Pythia_400to600, Pythia_600to800, Pythia_800to1200, Pythia_1200to2500, Pythia_2500toInf]
    lumi = 1
    xsecs = [xsecs_D.DYJetsToLL]*11
    labels = ['HT 0-70', 'HT 70-100', 'HT 100-200', 'HT 200-400', 'HT 400-600', 'HT 600-800', 'HT 800-1200', 'HT 1200-2500', 'HT 2500-Inf']
    dataname = 'Inclusive Pythia'

    plotKin(data, lumi, MCs, xsecs, labels,
            'HJet', 'Jpt',
            logx=True, logy=True,
            normToLumi=False,
            stack=True,
            density=False,
            show=show,
            done=True,
            folder=os.path.join(folder, "HTbinned"))

    plotKin(data, lumi, [Pythia_inclusive], xsecs, ['Inclusive Pythia'],
            'HJet', 'Jpt',
            logx=True, logy=True,
            normToLumi=False,
            stack=True,
            density=False,
            show=show,
            done=True,
            folder=os.path.join(folder, "Pythia"))

    plotKin(data, lumi, [Herwig_inclusive], xsecs, ['Inclusive Herwig'],
            'HJet', 'Jpt',
            logx=True, logy=True,
            normToLumi=False,
            stack=True,
            density=False,
            show=show,
            done=True,
            folder=os.path.join(folder, "Herwig"))

    #plotAllKin(baseline, lumi, MCs, xsecs, labels,
    #        normToLumi=False,
    #        stack=True,
    #        density=False,
    #        dataname=dataname)

    '''
    Compare summed HT histograms
    '''
    Pythia_inclusive = SAMPLE_LIST.lookup("DYJetsToLL").get_hist("Kin", tags)
    Herwig_inclusive = SAMPLE_LIST.lookup("DYJetsToLL_Herwig").get_hist("Kin", tags)
    Pythia_HT = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Kin", tags)

    baseline = None
    MCs = [Herwig_inclusive, Pythia_HT, Pythia_inclusive]

    lumi = 1
    xsecs = [xsecs_D.DYJetsToLL]*3

    labels = ['Inclusive Herwig', 'HT-binned Pythia', "Inclusive Pythia"]
    dataname = 'Inclusive Pythia'

    plotAllKin(baseline, lumi, MCs, xsecs, labels,
                   normToLumi=False,
                   stack=False,
                   density=True,
                   dataname=dataname,
                   show=show,
                   done=True,
                   folder=os.path.join(folder, "Compare"))

#doHTval("plots/HTvalidation")
