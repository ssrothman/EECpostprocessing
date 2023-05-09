from matplotlib import pyplot as plt
import numpy as np
import awkward as ak

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
