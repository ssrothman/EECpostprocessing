from matplotlib import pyplot as plt
import numpy as np
import awkward as ak


def plotJetValue(hdict, var, PU=None, etabin=None, ptbin=None, show=True, savefig=None):
    h = hdict['jets'][var]
    if PU is not None:
        h = h[{"PU" : PU}]
    if etabin is not None:
        h = h[{"eta" : etabin}]
    if ptbin is not None:
        h = h[{"pt" : ptbin}]

    h1d = h.project(var)
    vals = h1d.values(flow=False)
    errs = np.sqrt(h1d.variances(flow=False))
    xs = h.axes[var].centers
    widths = h.axes[var].widths

    plt.errorbar(xs, vals, yerr=errs, xerr=widths/2, fmt='o')
    plt.ylabel("Events")
    plt.xlabel(h.axes[var].label)

    plt.axvline(x=1, color='black', linestyle='--')

    if savefig is not None:
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
        if not show:
            plt.clf()
    if show:
        plt.show()


def plotJets(h, var, PU=-1, ylabel="# [A.U.]", savefig=None, clear=False, show=True, label=None):
    if PU == -1:
        hpu = h
    else:
        hpu = h[{"pu" : PU}]
    h1d = hpu.project(var)

    vals = h1d.values(flow=False)
    variances = h1d.variances(flow=False)

    if h.axes[var].transform is not None:
        plt.xscale('log')

    plt.ylabel(ylabel)
    plt.xlabel(h.axes[var].label)
    plt.errorbar(h.axes[var].centers, vals, yerr=np.sqrt(variances), fmt='o--', label=label)

    if savefig is not None:
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
    if show:
        plt.show()
    if clear:
        plt.clf()
