import numpy as np
import matplotlib.pyplot as plt

x = None

def set_hists(m):
    global x
    x = m.hists

def plotpure(name, label):
    S = x[name]['HtransS'][{'ptGen' : slice(3,None,sum)}].values(flow=True)
    S = np.nan_to_num(S/np.sum(S,axis=(0,1)))
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
    F = x[name]['HtransFG'][{'ptGen' : slice(3,None,sum)}]
    F.project(axis).plot(label=label)
    plt.title("factor")

def forward(name, factorside, EEC_from=None):
    if EEC_from is None:
        EEC_from = name

    if factorside not in ['Gen', 'Reco']:
        raise ValueError("factorside must be Gen or Reco")

    if factorside=='Gen':
        F = x[name]['HtransFG'].project('ptGen', 'dRbinGen').values(flow=True)
        S = x[name]['HtransSR'].values(flow=True)
        S = np.nan_to_num(S/np.sum(S, axis=(0,1)))
        T = np.einsum('ijkl,kl->ijkl', S, F)
    else:
        F = x[name]['HtransFR'].project('ptReco', 'dRbinReco').values(flow=True)
        S = x[name]['HtransSG'].values(flow=True)
        S = np.nan_to_num(S/np.sum(S, axis=(0,1)))
        T = np.einsum('ij,ijkl->ijkl', F, S)

    G = (x[EEC_from]['Hgen'] - x[EEC_from]['HgenUNMATCH']).values(flow=True)

    transfered = np.einsum('ijkl,kl->ij', T, G)

    R = (x[EEC_from]['Hreco'] - x[EEC_from]['HrecoPUjets'] 
         - x[EEC_from]['HrecoUNMATCH']).values(flow=True)

    return transfered, R

def plotforward(name, factorside, EEC_from=None, title=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1])

    if title is not None:
        fig.suptitle(title)

    transfered, R = forward(name, factorside, EEC_from)

    t = np.sum(transfered, axis=0)
    r = np.sum(R, axis=0)
    ratio = t/r
    ax0.scatter(np.arange(52), t, label='Forward transfered')
    ax0.scatter(np.arange(52), r, label='Background-subtracted reco')
    ax0.set_ylabel("EEC value")
    ax0.legend()

    ax1.scatter(np.arange(52), ratio)
    ax1.set_ylabel("Ratio")
    ax1.set_xlabel("Reco delta-R bin")
    ax1.axhline(1, color='k', linestyle='--')

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
