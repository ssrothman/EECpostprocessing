import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import hist
import mplhep as hep
from plotting.util import *

from samples.latest import SAMPLE_LIST

def plotMatchRate(data, wrt='pt', pid=None):
    fig, ax = setup_plain()

    add_cms_info(ax, True)

    if pid is not None:
        H = data['partmatch'][{'partSpecies': pid}]
        if pid==0:
            text = "Tracks"
        elif pid==1:
            text = "Photons"
        elif pid==2:
            text = "Neutral Hadrons"

        ax.text(0.7, 0.1, text, transform=ax.transAxes)
    else:
        H = data['partmatch']

    Hproj = H[{'eta' : 0}].project(wrt, 'nMatch')
    vals = Hproj.values()
    errs = np.sqrt(Hproj.variances())

    pmatch = 1-vals[:,0]/vals.sum(axis=1)

    properr2 = np.square(errs/vals)
    pmatch_err = pmatch*np.sqrt(np.sum(properr2, axis=1))

    centers = Hproj.axes[0].centers
    edges = Hproj.axes[0].edges
    widths = Hproj.axes[0].widths

    ax.errorbar(centers, pmatch,
                yerr=pmatch_err, 
                xerr=widths/2, 
                fmt='o', color='black')
    ax.set_xlabel(Hproj.axes[0].label)
    ax.set_ylabel("Particle matching rate")
    ax.set_ylim(0,1.1)
    ax.axhline(1.0, color='red', linestyle='--')

    plt.show()

def plotMatchFrac(data, wrt):
    fig, ax = setup_plain()

    add_cms_info(ax, True)

    H = data['jetmatch']
    Hproj = H.project(wrt, 'fracMatched')

    vals = Hproj.values()
    errs = np.sqrt(Hproj.variances())

    fracCenters = Hproj.axes[1].centers
    means = np.sum(fracCenters[None,:]*vals, axis=1)/np.sum(vals, axis=1)
    print(vals[3,:])
    print(fracCenters)
    print(vals[3,:]*fracCenters)
    print(means[3])
    eacherr = np.square(fracCenters[None,:] * errs)
    errs = np.sqrt(np.sum(eacherr, axis=1))/np.sum(vals, axis=1)

    centers = Hproj.axes[0].centers
    edges = Hproj.axes[0].edges
    widths = Hproj.axes[0].widths

    ax.errorbar(centers, means,
                yerr=errs,
                xerr=widths/2,
                fmt='o', color='black')
    ax.set_xlabel(Hproj.axes[0].label)
    ax.set_ylabel("Fraction of jet pT that is matched")
    ax.set_ylim(0,1.1)
    ax.axhline(1.0, color='red', linestyle='--')

    plt.show()

def plotResolution(data, species, whichax, wrt, binMin, binMax):
    fig, ax = setup_plain()

    add_cms_info(ax, True)

    if species == 'JET':
        text = 'Jets'
    elif species == 'EM0':
        text = 'Photons'
    elif species == 'HAD0':
        text = 'Neutral Hadrons'
    elif species == 'TRK':
        text = 'Tracks'

    ax.text(0.7, 0.1, text, transform=ax.transAxes)

    H = data['res'][species][whichax]
    Hproj = H.project(wrt, whichax)
    Hproj = Hproj[{wrt : slice(binMin, binMax, hist.sum)}]

    Hproj.plot(density=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--')

    plt.show()
    

S = SAMPLE_LIST.lookup("DYJetsToLL_allHT")
Hdict = S.get_hist('Match')

#plotResolution(Hdict, 'TRK', 'dpt', 'partPt', 5, 10)
#plotResolution(Hdict, 'HAD0', 'dpt', 'partPt', 5, 10)
#plotResolution(Hdict, 'EM0', 'dpt', 'partPt', 5, 10)
#plotResolution(Hdict, 'JET', 'dpt', 'partPt', 5, 10)
#
#plotMatchFrac(Hdict, 'Jpt')
#
#plotMatchRate(Hdict, 'eta')
#plotMatchRate(Hdict, 'DRaxis')
#plotMatchRate(Hdict, 'btag')
#plotMatchRate(Hdict, 'partSpecies')

plotMatchRate(Hdict, 'partPt', pid=0)
plotMatchRate(Hdict, 'partPt', pid=1)
plotMatchRate(Hdict, 'partPt', pid=2)
plotMatchRate(Hdict, 'Jpt')
