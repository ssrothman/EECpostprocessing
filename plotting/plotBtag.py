import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import mplhep as hep
import seaborn as sns

from plotting.util import *

plt.style.use(hep.style.CMS)

def plotBmatch(data):
    fig, ax = setup_plain()
    add_cms_info(ax, True)

    values = data['bmatch'].project('pt', 'NumBMatch').values()

    print(values.shape)
    pfail = values[:, 0]/np.sum(values, axis=1)
    print(pfail.shape)

    edges = np.asarray(config.binning.bins.Beffpt)
    centers = (edges[1:] + edges[:-1])/2
    widths = edges[1:] - edges[:-1]

    ax.errorbar(centers, 1-pfail, xerr=widths/2, fmt='o')
    ax.axhline(1.0, c='k', ls='--')
    ax.axhline(0.0, c='k', ls='--')
    ax.set_xscale('log')
    ax.set_ylabel("AK8-AK4 matching efficiency")
    ax.set_xlabel("AK8 jet $p_T$ [GeV]")
    plt.show()

def plotBtag(data, how='purity', wp='tight'):
    fig, ax = setup_plain()
    add_cms_info(ax, True)

    values = data['btag'].project("btag_%s"%wp, 'genflav').values()
    if how == 'purity':
        values/=np.sum(values, axis=1, keepdims=True)
    elif how == 'efficiency':
        values/=np.sum(values, axis=0, keepdims=True)


    sns.heatmap(values, cmap="Reds", square=False, annot=True, cbar=False, ax=ax)

    ax.set_ylabel('B tag')
    ax.set_xlabel('Gen flavor')
    ax.set_xticklabels(['udsg', 'c', 'b'])
    ax.set_yticklabels(["Fail", "Pass"])

    savefig("btag/%s_%s.png"%(wp, how))

    plt.show()

def plotBtagEffPerPt(data, wp='tight', iflav=0, flavname='udsg', mode = 'pass'):
    fig, ax = setup_plain()
    add_cms_info(ax, True)

    values = data['btag'].project('pt', 'eta', "btag_%s"%wp, 'genflav').values()
    errs = np.sqrt(data['btag'].project('pt', 'eta', "btag_%s"%wp, 'genflav').variances())

    N = np.sum(values, axis=2, keepdims=True)
    values/=N
    errs/=N

    ax.set_ylabel("%s-jet tag %s rate" % (flavname, mode))
    ax.set_xlabel("Jet $p_T$ [GeV]")
    
    xs, xerrs = getAXcenters_errs('Beffpt')
    tag = 0 if mode == 'fail' else 1
    for eta, etaname in zip(range(2), ['Barrel', 'Endcaps']):
        if eta==0:
            ax.errorbar(xs*0.99, values[:, eta, tag, iflav], 
                        yerr=errs[:, eta, tag, iflav],
                        xerr=xerrs,
                        label=etaname, fmt='o', capsize=3)
        else:
            ax.errorbar(xs*1.001, values[:, eta, tag, iflav], 
                        yerr=errs[:, eta, tag, iflav],
                        xerr=xerrs,
                        label=etaname, fmt='o', capsize=3)


    ax.axhline(1.0, color='k', linestyle='--')
    ax.axhline(0.0, color='k', linestyle='--')

    plt.xscale('log')

    ax.legend()
    savefig("btag/%s_%s_%s.png"%(wp, flavname, mode))
    plt.show()

def plotBtagPurePerPt(data, wp='tight', ieta=0, mode='pass'):
    fig, ax = setup_plain()
    add_cms_info(ax, True)

    values = data['btag'].project('pt', 'eta', "btag_%s"%wp, 'genflav').values()
    errs = np.sqrt(data['btag'].project('pt', 'eta', "btag_%s"%wp, 'genflav').variances())

    N = np.sum(values, axis=3, keepdims=True)
    values/=N
    errs/=N

    ax.set_ylabel("b-tag %s region composition" % mode)
    ax.set_xlabel("Jet $p_T$ [GeV]")

    xs, xerrs = getAXcenters_errs('Beffpt')
    edges = config.binning.bins.Beffpt

    tag = 0 if mode == 'fail' else 1
    nextvals = np.zeros_like(values[:, ieta, tag, 0])

    for iflav, flavname in zip(range(3), ['udsg', 'c', 'b']):
        thesevals = values[:, ieta, tag, iflav]

        ax.stairs(thesevals+nextvals, edges, baseline = nextvals, fill=True, label = flavname)
        nextvals += thesevals

    plt.legend(facecolor='white', edgecolor='black', frameon=True, framealpha=1, fancybox=False)
    plt.xscale('log')
    savefig("btag/%s_%s_%s_eta%d.png"%(wp, mode, 'pure', ieta))
    plt.show()
