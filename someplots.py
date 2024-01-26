from plotting.EECutil import EEC
from plotting.plotEEC import *

pythia = EEC('Nov12_2023_pythia/EEC/hists.pkl')
herwig = EEC('Nov12_2023_herwig/EEC/hists.pkl')

def compareShape(name):
    plotEEC(pythia, 'CaloShareEEC1', name, 
            ptbin=3, etabin=None, pubin=None,
            density=True, label='Pythia')
    plotEEC(herwig, 'CaloShareEEC1', name,
            ptbin=3, etabin=None, pubin=None,
            density=True, label='Herwig')
    plt.show()

def compareRatio(name, to):
    compareEECratio([pythia, herwig], ['CaloShareEEC1']*2, [name]*2, 
                    labels=['Pythia', 'Herwig'],
                    ptbins=[3]*2, etabins=[None]*2, pubins=[None]*2,
                    ratio_to=to, density=False, ratio_mode='difference')

def compareRatioPU(name, to):
    compareEECratio([pythia]*2, ['CaloShareEEC1']*2, [name]*2,
                    labels= ['Low NPU', 'High NPU'],
                    ptbins=[3]*2, etabins=[None]*2, pubins=[0, 4],
                    ratio_to=to, density=False, ratio_mode=None)

def compareTotalPU(name):
    Nlow = pythia.Hdict['CaloShareEEC1']['Hreco'][{'nPU' : 0}].sum()
    Nhigh = pythia.Hdict['CaloShareEEC1']['Hreco'][{'nPU' : 4}].sum()
    plotEEC(pythia, 'CaloShareEEC1', name,
            ptbin=3, etabin=None, pubin=0,
            density=Nlow, label='Low NPU Bin')
    plotEEC(pythia, 'CaloShareEEC1', name,
            ptbin=3, etabin=None, pubin=4,
            density=Nhigh, label='High NPU Bin')
    plt.show()

def factors():
    compareFactors([pythia]*2, ['NaiveEEC']*2, ['Low PU', 'High PU'],
                   [3]*2, [None]*2, [0,3])

    compareFactors([pythia]*2, ['NaiveEEC']*2, ['Central', 'Forward'],
                   [3]*2, [0,3], [None]*2)

def ofrward():
    compareForward(pythia, 'CaloShareEEC1', herwig, 'CaloShareEEC1', 3, None, None, 
                   mode='ratio', doTemplates=True, 
                   folder='plots_nov15/forward_herwig/templated')
    compareForward(pythia, 'CaloShareEEC1', pythia, 'CaloShareEEC2', 3, None, None, 
                   mode='ratio', doTemplates=True, 
                   folder='plots_nov15/forward_stat/templated')
    compareForward(pythia, 'CaloShareEEC1', pythia, 'CaloShareEEC2', 3, None, None, 
                   mode='ratio', doTemplates=False, 
                   folder='plots_nov15/forward_stat/notemplate')


def compareRatios():
    compareEECratio([pythia]*4, ['CaloShareEEC']*4, ['HrecoPUjets']*4, 
                    ['PU bin %d'%i for i in range(4)],
                    [3]*4, [None]*4, [i for i in range(4)], 
                    'Hreco', density=False, ratio_mode='difference',
                    folder = 'plots_nov15/PUjets/PU')

    compareEECratio([pythia]*2, ['CaloShareEEC']*2, ['HrecoPUjets']*2, 
                    ['Central', 'Forward'],
                    [3]*2, [0,3], [None]*2, 'Hreco', density=False, ratio_mode='ratio',
                    folder = 'plots_nov15/PUjets/eta')

    compareEECratio([pythia]*4, ['CaloShareEEC']*4, ['HrecoUNMATCH']*4, 
                    ['PU bin %d'%i for i in range(4)],
                    [3]*4, [None]*4, [i for i in range(4)], 
                    'Hreco', density=False, ratio_mode='difference',
                    folder = 'plots_nov15/UNMATCH/PU')

    compareEECratio([pythia]*2, ['CaloShareEEC']*2, ['HrecoUNMATCH']*2, 
                    ['Central', 'Forward'],
                    [3]*2, [0,3], [None]*2, 'Hreco', density=False, ratio_mode='ratio',
                    folder = 'plots_nov15/UNMATCH/eta')

def comparePurityEta():
    plotPurityStability(pythia, 'CaloShareEEC', None, 0, None, True, label='Central')
    plotPurityStability(pythia, 'CaloShareEEC', None, 3, None, True, label='Forward')
    plt.savefig("plots_nov15/purity_eta.png", format='png', bbox_inches='tight')
    plt.show()
    plotPurityStability(pythia, 'CaloShareEEC', None, None, 0, True, label='Low PU')
    plotPurityStability(pythia, 'CaloShareEEC', None, None, 3, True, label='High PU')
    plt.savefig("plots_nov15/purity_PU.png", format='png', bbox_inches='tight')
    plt.show()

#ofrward()
#compareRatio()
#comparePurityEta()
