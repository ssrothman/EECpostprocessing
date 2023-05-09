import hist
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

def plotMatchRate(h, var, nmatch = -1, ylabel='Particle matching rate', savefig=None, clear=False, show=True, match='nmatch', label=None):
    h2d = h.project('nmatch', var)
    vals = h2d.values(flow=True)
    variances = h2d.variances(flow=True)
    

    if h.axes[var].traits.underflow:
        vals = vals[:,1:]
        variances = variances[:,1:]
    if h.axes[var].traits.overflow:
        vals = vals[:,:-1]
        variances = variances[:,:-1]

    norms = np.sum(vals, axis=0)

    if(nmatch ==- 1):
        pmiss = 1-vals[0, :]/norms
        dpmiss = np.sqrt(variances[0, :])/norms
    else:
        pmiss = vals[nmatch, :]/norms
        dpmiss = np.sqrt(variances[nmatch, :])/norms


    varaxis = h.axes[var]

    plt.ylabel(ylabel)
    plt.ylim(0, 1.1)
    plt.axhline(y=1, color='black', linestyle='--')
    if(type(varaxis) == hist.axis.Regular):
        x = varaxis.centers
        if varaxis.transform is not None:
            plt.xscale('log')
        
        plt.errorbar(x, pmiss, yerr=dpmiss, fmt='o--', label=label)
        plt.xlabel(h.axes[var].label)
    elif type(varaxis) == hist.axis.IntCategory:
        x = np.arange(varaxis.size)
        plt.errorbar(x, pmiss, yerr=dpmiss, fmt='o', label=label)
        plt.xticks(x, varaxis.value(x))
        plt.xlabel(h.axes[var].label)
    else:
        raise ValueError("Unknown axis type")

    if savefig is not None:
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
    if show:
        plt.show()
    if clear:
        plt.clf()

