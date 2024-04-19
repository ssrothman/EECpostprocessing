import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import mplhep as hep
from plotting.util import *
import hist

plt.style.use(hep.style.CMS)

from samples.latest import SAMPLE_LIST

def plotDataMC(data, mcs, mclabels=['MC'], bins={}, which='DR', name=None):
    singlefun = plotDR_single if which == 'DR' else plotTurnon_single
    partptbin = None if which == 'DR' else int(which[-1])

    fig, (ax0, ax1) = setup_ratiopad()
    add_cms_info(ax0, False)

    txt = binnedtext(bins)
    ax0.text(0.05, 0.95, txt, transform=ax0.transAxes, 
             verticalalignment='top')

    _, _, dataratio, dataratioerr = singlefun(
            data, 'k', 'Data', ax0, bins=bins,
            partptbin = partptbin)

    if isinstance(mcs, dict):
        mcs = [mcs]

    for mc, label in zip(mcs, mclabels):
        x, xerr, mcratio, mcratioerr = singlefun(mc, None, label, ax0, 
                                                 bins=bins,
                                                 partptbin=partptbin)

        ratioratio = dataratio/mcratio
        ratioratioerr = ratioratio*np.sqrt((dataratioerr/dataratio)**2 + (mcratioerr/mcratio)**2)
        ax1.errorbar(x, ratioratio, 
                     xerr=xerr,
                     yerr=ratioratioerr, 
                     fmt='o', color=None)
        
    ax1.axhline(y=1, color='k', linestyle='--')
    ax0.legend(loc='upper left', bbox_to_anchor=(0.05,0.85))

    if which == 'DR':
        ax1.set_xlabel("$\Delta R$")
        ax0.set_ylabel("N(Charged)/N(Neutral)")
    else:
        ax1.set_xlabel("Jet $p_T$ [GeV]")
        if partptbin == 0:
            ax0.set_ylabel("N($p_T > 10$ GeV)/N($p_T < 1$ GeV")
        elif partptbin == 1:
            ax0.set_ylabel("N($p_T > 10$ GeV)/N($1 < p_T < 2$ GeV")
        elif partptbin == 2:
            ax0.set_ylabel("N($p_T > 10$ GeV)/N($2 < p_T < 3$ GeV")
        elif partptbin == 3:
            ax0.set_ylabel("N($p_T > 10$ GeV)/N($3 < p_T < 5$ GeV")
        elif partptbin == 4:
            ax0.set_ylabel("N($p_T > 10$ GeV)/N($5 < p_T < 10$ GeV")

    ax1.set_ylabel("Data/MC")

    if name is not None:
        savefig('mult/%s.png'%name)

    plt.show()

def plotDR_single(datadict, color, label, ax, bins={}, partptbin=None):
    H = datadict['H'][{'partPtCategory' : slice(1,None,hist.sum)}]
    H = apply_bins(H, bins)
    H = H.project('partSpecies', "DRaxis")

    neu = H[{'partSpecies' : slice(1,None,hist.sum)}]
    chg = H[{'partSpecies' : 0}]

    neuvals = neu.values()
    neuerrs = np.sqrt(neu.variances())

    chgvals = chg.values()
    chgerrs = np.sqrt(chg.variances())

    ratio = chgvals/neuvals
    ratioerr = ratio*np.sqrt((chgerrs/chgvals)**2 + (neuerrs/neuvals)**2)

    centers = neu.axes[0].centers
    widths = neu.axes[0].widths

    ax.errorbar(centers, ratio, xerr=widths/2, yerr=ratioerr, fmt='o',
                color=color, label=label)

    ax.set_xscale('log')

    return centers, widths/2, ratio, ratioerr

def plotTurnon_single(datadict, color, label, ax, bins={}, 
                      partptbin=0):
    H = apply_bins(datadict['H'], bins)
    H = H.project('pt', 'partPtCategory')

    high = H[{'partPtCategory' : 5}]
    low = H[{'partPtCategory' : partptbin}]

    highvals = high.values()
    higherrs = np.sqrt(high.variances())

    lowvals = low.values()
    lowerrs = np.sqrt(low.variances())

    ratio = highvals/lowvals
    ratioerr = ratio*np.sqrt((higherrs/highvals)**2 + (lowerrs/lowvals)**2)

    centers = low.axes[0].centers
    widths = low.axes[0].widths

    ax.errorbar(centers, ratio, xerr=widths/2, yerr=ratioerr, fmt='o',
                color=color, label=label)

    ax.set_xscale('log')

    return centers, widths/2, ratio, ratioerr

SAMPLE_LIST.lookup('DYJetsToLL_allHT').load_hist('Multiplicity')
SAMPLE_LIST.lookup('DYJetsToLL_Herwig').load_hist('Multiplicity')
SAMPLE_LIST.lookup('DATA_2018UL').load_hist("Multiplicity")

herwig = SAMPLE_LIST.lookup("DYJetsToLL_Herwig").get_hist("Multiplicity")
pythia = SAMPLE_LIST.lookup("DYJetsToLL_allHT").get_hist("Multiplicity")
data = SAMPLE_LIST.lookup("DATA_2018UL").get_hist("Multiplicity")

plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 0, 'eta' : 0}, 
           which='Turnon0',
           name = 'TrkTurnon0')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 0, 'eta' : 0}, 
           which='Turnon1',
           name = 'TrkTurnon1')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 0, 'eta' : 0}, 
           which='Turnon2',
           name = 'TrkTurnon2')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 0, 'eta' : 0}, 
           which='Turnon3',
           name = 'TrkTurnon3')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 0, 'eta' : 0}, 
           which='Turnon4',
           name = 'TrkTurnon4')

plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 1, 'eta' : 0}, 
           which='Turnon0',
           name = 'PhoTurnon0')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 1, 'eta' : 0}, 
           which='Turnon1',
           name = 'PhoTurnon1')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 1, 'eta' : 0}, 
           which='Turnon2',
           name = 'PhoTurnon2')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 1, 'eta' : 0}, 
           which='Turnon3',
           name = 'PhoTurnon3')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 1, 'eta' : 0}, 
           which='Turnon4',
           name = 'PhoTurnon4')

plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 2, 'eta' : 0}, 
           which='Turnon0',
           name = 'HadTurnon0')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 2, 'eta' : 0}, 
           which='Turnon1',
           name = 'HadTurnon1')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 2, 'eta' : 0}, 
           which='Turnon2',
           name = 'HadTurnon2')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 2, 'eta' : 0}, 
           which='Turnon3',
           name = 'HadTurnon3')
plotDataMC(data, [pythia], ['Pythia'], 
           bins={'partSpecies' : 2, 'eta' : 0}, 
           which='Turnon4',
           name = 'HadTurnon4')

#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], 
#           bins={'partSpecies' : 0, 'eta' : 1},
#           which='Turnon')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], 
#           bins={'partSpecies' : 0, 'eta' : 2}, 
#           which='Turnon')

#plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 0}, 
#           which = 'DR')
#plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 1}, 
#           which = 'DR')
#plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 2}, 
#           which = 'DR')
plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 3, 'eta' : 0}, 
           which = 'DR',
           name = 'DReta0')
plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 3, 'eta' : 1}, 
           which = 'DR',
           name = 'DReta1')
plotDataMC(data, [pythia], ['Pythia'], bins={'pt' : 3, 'eta' : 2}, 
           which = 'DR',
           name = 'DReta2')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 1, 'eta' : 0}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 1, 'eta' : 1}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 1, 'eta' : 2}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 2, 'eta' : 0}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 2, 'eta' : 1}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 2, 'eta' : 2}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 3, 'eta' : 0}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 3, 'eta' : 1}, which='DR')
#plotDataMC(data, [pythia, herwig], ['Pythia', 'Herwig'], bins={'pt' : 3, 'eta' : 2}, which='DR')
