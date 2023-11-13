from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import reading.reader

#f = '../CMSSW_10_6_26/src/SRothman/NANO_NANO_NANO.root'
#f = 'root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Oct19_2023_EECtest_take2/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/231019_192658/0000/NANO_NANO_NANO_99.root'
#f = 'root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Oct30_2023_EECcorr/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL/231030_200321/0000/NANO_NANO_NANO_1.root'
f = 'root://cmseos.fnal.gov//store/group/lpcpfnano/srothman/Nov01_2023_EECcorr_caloreco_fixed_fixed_herwig/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/DYJetsToLL/231102_011242/0000/NANO_NANO_NANO_1.root'
x = NanoEventsFactory.from_root(f).events()
print("GOT X")

from processing.EECProcessor import EECProcessor
p = EECProcessor(['NaiveEEC'], ['NaiveMatch'], 52, False, False, True)
q = p.process(x)

#print(q['EEC']['Htrans'].project("EECwtReco") - q['EEC']['HrecoPure'].project("EECwt")) 
print()
print("--"*20)
print()

rTransfer = reading.reader.transferreader(x, 'EECTransfer')
rGenEEC = reading.reader.EECreader(x, 'GenEEC')
rGenEECUNMATCH = reading.reader.EECreader(x, "GenEECUNMATCH")
rRecoEEC = reading.reader.EECreader(x, 'RecoEEC')
rRecoEECPU = reading.reader.EECreader(x, 'RecoEECPU')
rRecoJet = reading.reader.jetreader(x, '', 'SimonJets')
rGenJet = reading.reader.jetreader(x, '', 'GenSimonJets')

mask = np.abs(rRecoJet.simonjets.eta) < 1.0

import binning.binEEC_binwt
from importlib import reload

#print(q['EEC']['Htrans'].project("EECwtGen"))
#print(q['EEC']['HgenPure'].project("EECwt"))
#q['EEC']['HgenPure'] = q['EEC']['Hgen'] - q['EEC']['HgenUNMATCH']
#q['EEC']['HrecoPure'] = q['EEC']['Hreco'] - q['EEC']['HrecoPUjets'] - q['EEC']['HrecoUNMATCH']
#q['NaiveEEC']['HgenPure'] = q['NaiveEEC']['Hgen'] - q['NaiveEEC']['HgenUNMATCH']
#q['NaiveEEC']['HrecoPure'] = q['NaiveEEC']['Hreco'] - q['NaiveEEC']['HrecoPUjets'] - q['NaiveEEC']['HrecoUNMATCH']

#naiveS = q['NaiveEEC']['HtransSR'].values(flow=True)
#naiveS = np.nan_to_num(naiveS/np.sum(naiveS,axis=(0, 1)))
#naiveF = q['NaiveEEC']['HtransFR'].project('ptReco', 'dRbinReco').values(flow=True)
#naiveG = q['NaiveEEC']['HgenPure'].values(flow=True)
#naiveR = q['NaiveEEC']['HrecoPure'].values(flow=True)
#naiveSG = np.einsum('ijkl,kl->ij', naiveS, naiveG)
#naiveFG = naiveF * naiveG
#naiveFSG = naiveF*np.einsum('ijkl,kl->ij', naiveS, naiveG)

#naiveF2 = q['NaiveEEC']['HtransFG'].project('ptGen', 'dRbinGen').values(flow=True)
#naiveSFG = np.einsum('ijkl,kl->ij', naiveS, naiveF2*naiveG)

#fancyS = q['EEC']['HtransS'].values(flow=True)
#fancyS = np.nan_to_num(fancyS/np.sum(fancyS,axis=(0, 1)))
#fancyF = q['EEC']['HtransFR'].project('ptReco', 'dRbinReco').values(flow=True)
#fancyG = q['EEC']['HgenPure'].values(flow=True)
#fancyR = q['EEC']['HrecoPure'].values(flow=True)
#fancySG = np.einsum('ijkl,kl->ij', fancyS, fancyG)
#fancyFSG = fancyF*np.einsum('ijkl,kl->ij', fancyS, fancyG)

#fancyF2 = q['EEC']['HtransFG'].project('ptGen', 'dRbinGen').values(flow=True)
#fancySFG = np.einsum('ijkl,kl->ij', fancyS, fancyF2*fancyG)

def plots():
    x = np.arange(52)
    fancy = np.sum(fancyS, axis=(0,2))
    fancy0 = np.diag(fancy)/np.sum(fancy, axis=0)
    fancy1 = np.diag(fancy)/np.sum(fancy, axis=1)
    naive = np.sum(naiveS, axis=(0,2))
    naive0 = np.diag(naive)/np.sum(naive, axis=0)
    naive1 = np.diag(naive)/np.sum(naive, axis=1)
    plt.title("axis 0")
    plt.scatter(x, fancy0, label='fancy')
    plt.scatter(x, naive0, label='naive')
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    plt.title("axis 1")
    plt.scatter(x, fancy1, label='fancy')
    plt.scatter(x, naive1, label='naive')
    plt.ylim(0,1)
    plt.legend()
    plt.show()
    plt.title("factors")
    q['EEC']['HtransFR'].project("dRbinGen").plot(label='fancy')
    q['FancyNaiveEEC']['HtransFR'].project("dRbinGen").plot(label='naive')
    plt.axhline(1, color='black', linestyle='--')
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

#plots()

def test(name):
    recoH = q[name]['HrecoPure']
    genH = q[name]['HgenPure']
    transrecoH = q[name]['Htrans'].project('ptReco', 'dRbinReco', 'EECwtReco')
    transgenH = q[name]['Htrans'].project('ptGen', 'dRbinGen', 'EECwtGen')

    recovals = recoH.values(flow=True)
    genvals = genH.values(flow=True)
    transrecovals = transrecoH.values(flow=True)
    transgenvals = transgenH.values(flow=True)

    print('--'*20)
    print(name)
    print("(reco==0) & (transreco!=0)", np.sum((recovals==0) & (transrecovals!=0)))
    print("(reco!=0) & (transreco==0)", np.sum((recovals!=0) & (transrecovals==0)))
    print("Max reco difference", np.max(np.abs(recovals - transrecovals)))
    print("target[1,0,30]", recovals[1,0,30])
    print("actual[1,0,30]", transrecovals[1,0,30])
    print("(gen==0) & (transgen!=0)", np.sum((genvals==0) & (transgenvals!=0)))
    print("(gen!=0) & (transgen==0)", np.sum((genvals!=0) & (transgenvals==0)))
    print()

#test('EEC')
#test('NaiveEEC')
def test2(i):
    fancysum = q['EEC']['HtransFR'][{'EECwtGen':i}].project('factor').sum(flow=True)
    naivesum = q['NaiveEEC']['HtransFR'][{'EECwtGen':i}].project('factor').sum(flow=True)
    (q['EEC']['HtransFR'][{'EECwtGen' : i}].project('factor')/fancysum).plot(
            label='Fancy')
    (q['NaiveEEC']['HtransFR'][{'EECwtGen' : i}].project('factor')/naivesum).plot(
            label='Naive')
    plt.axvline(1.0, color='k', linestyle='--')
    plt.legend()
    plt.show()
