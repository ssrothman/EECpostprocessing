import numpy as np
import matplotlib.pyplot as plt

x = None

def set_hists(m):
    global x
    x = m.hists

def plotpure(name, label):
    S = x[name]['HtransS'][{'ptGen' : slice(3,None,sum)}].values(flow=True)
    S = np.nan_to_num(S/np.sum(S,axis=(0,1)))
    print(S.shape)
    xval = np.arange(52)
    a = np.sum(S, axis=(0))
    plt.title("pure")
    plt.scatter(xval, np.diag(a)/np.sum(a, axis=0), label=label)

def plotstab(name, label):
    S = x[name]['HtransS'][{'ptGen' : slice(3,None,sum)}].values(flow=True)
    S = np.nan_to_num(S/np.sum(S,axis=(0,1)))
    xval = np.arange(52)
    a = np.sum(S, axis=(0))
    plt.title("stab")
    plt.scatter(xval, np.diag(a)/np.sum(a, axis=1), label=label)

def plotfactor(name, label, axis='dRbinGen'):
    F = x[name]['HtransF'][{'ptGen' : slice(3,None,sum)}]
    F.project(axis).plot(label=label)
    plt.title("factor")

def plotforward(name):
    F = x[name]['HtransF'].project('ptGen', 'dRbinGen').values(flow=True)
    S = x[name]['HtransS'].values(flow=True)
    S = np.nan_to_num(S/np.sum(S,axis=(0,1)))
    G = (x[name]['Hgen'] - x[name]['HgenUNMATCH']).values(flow=True)
    R = (x[name]['Hreco'] - x[name]['HrecoPUjets'] 
         - x[name]['HrecoUNMATCH']).values(flow=True)
    
    forward = np.einsum('ijkl,kl->ij', S, F*G)

    plt.scatter(np.arange(52), np.sum(forward, axis=0)/np.sum(R, axis=0))
    plt.ylim(0.999,1.001)
    plt.ylabel("Forward-transfered / Reco")
    plt.xlabel("Reco delta-R bin")
    plt.tight_layout()
    plt.show()

def plots_fancynaive():
    plotpure('FancyEEC', 'fancy')
    plotpure('NaiveEEC', 'naive')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    plotstab('FancyEEC', 'fancy')
    plotstab('NaiveEEC', 'naive')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    plotfactor('FancyEEC', 'fancy')
    plotfactor('NaiveEEC', 'naive')
    plt.axhline(1, color='black', linestyle='--')
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

    plotfactor('FancyEEC', 'fancy', 'EECwtGen')
    plotfactor('NaiveEEC', 'naive', 'EECwtGen')
    plt.axhline(1, color='black', linestyle='--')
    plt.ylim(0, 4)
    plt.xscale('log')
    plt.legend()
    plt.show()

def plots_corr():
    plotpure('FancyCorrEEC', 'fancyCorr')
    plotpure('FancyEEC', 'fancy')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    plotstab('FancyCorrEEC', 'fancyCorr')
    plotstab('FancyEEC', 'fancy')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    plotfactor('FancyCorrEEC', 'fancyCorr')
    plotfactor('FancyEEC', 'fancy')
    plt.axhline(1, color='black', linestyle='--')
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
