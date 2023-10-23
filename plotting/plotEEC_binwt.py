import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

import util.EECutil

import binning.binEEC_binwt

def load_proj(path):
    values = np.memmap(path, dtype=np.float64,
                       mode='c', shape=(12, 52, 52))
    return values

def load_transfer(path):
    values = np.memmap(path, dtype=np.float64,
                       mode='c', shape=(12,52,52,12,52,52))
    return values

def setup_Hdict(basepath):
    Hdict = {}
    for name in os.listdir(basepath):
        Hdict[name ] = {}
        for key in os.listdir(os.path.join(basepath, name)):
            if 'trans' in key:
                Hdict[name][key[:-4]]=load_transfer(
                        os.path.join(basepath,name,key))
            else:
                Hdict[name][key[:-4]]=load_proj(
                        os.path.join(basepath,name,key))
    return Hdict


edges = np.linspace(0, 0.5, 51)
edges[0] = 1e-10
dRaxis = hist.axis.Variable(edges, name='dR', label='$\Delta R$')

wtaxis = hist.axis.Regular(50, 1e-6, 1, name='EECwt', label='EEC weight',
                           transform=hist.axis.transform.log)

ptedges = np.linspace(0, 500, 11)
ptedges = np.concatenate(([-1], ptedges, [1000]))
ptaxis = hist.axis.Variable(ptedges, name='pt', label='pt')

def plotValues(values, errs, xs, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    differences = xs[1:] - xs[:-1]
    differences = np.concatenate(([differences[0]], differences))
    ax.errorbar(xs, values, yerr=errs, xerr = differences/2, ms=3, fmt='o', label=label)
    if label is not None:
        ax.legend()

def applyPtBin(Hval, ptbin):
    if ptbin is not None:
        Hval = Hval[{'pt':ptbin}]

    Hval = Hval.project('dRbin', 'EECwt')

    return Hval

def plotEEC(val_mmap, ptbin=None, label=None, logwidth=True, ax=None):
    if ax is None:
        ax = plt.gca()

    if ptbin is not None:
        vals = val_mmap[ptbin, :, :]
    else:
        vals = np.sum(val_mmap, axis=0)

    EECwts = wtaxis.centers
    vals = np.sum(vals[:,1:-1] * EECwts[None, :], axis=-1)

    xs = dRaxis.centers
    edges = dRaxis.edges
    if logwidth:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    vals = vals[1:-1]/widths

    ax.set_xlabel("$\Delta R$")
    if logwidth:
        ax.set_ylabel("$\\frac{d\\sigma^{(2)}}{d\\log\\Delta R}$ [Unnormalized]")
    else:
        ax.set_ylabel("$\\frac{d\\sigma^{(2)}}{d\\Delta R}$ [Unnormalized]")

    plotValues(vals, 0, xs, label=label, ax=ax)

def plotRatio(val1, val2, ptbin=None, logwidth=True, 
              label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if ptbin is not None:
        val1 = val1[ptbin, :, :]
        val2 = val2[ptbin, :, :]
    else:
        val1 = np.sum(val1, axis=0)
        val2 = np.sum(val2, axis=0)

    EECwts = wtaxis.centers
    val1 = np.sum(val1[:,1:-1] * EECwts[None, :], axis=-1)[1:-1]
    val2 = np.sum(val2[:,1:-1] * EECwts[None, :], axis=-1)[1:-1]

    xs = dRaxis.centers
    edges = dRaxis.edges
    if logwidth:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    val1 = val1/widths
    val2 = val2/widths

    ratio = val1/val2

    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylabel("Ratio")
    ax.set_xlabel("$\Delta R$")
    plotValues(ratio, 0, xs, label=label, ax=ax)

def makeTransfer(Hdict, ptbin, includeInefficiency=True):
    trans = Hdict['Htrans']

    if includeInefficiency:
        gen = Hdict['Hgen']
    else:
        gen = Hdict['HgenPure']

    if ptbin is not None:
        gen = gen[ptbin, :, :]
        trans = trans[ptbin, :, :, ptbin, :, :]
    else:
        gen = np.sum(gen, axis=0)
        trans = np.sum(trans, axis=(0, 3))

    transValue = trans.copy()

    for i in range(trans.shape[2]):
        for j in range(trans.shape[3]):
            if gen[i,j] > 0:
                transValue[:,:,i,j]/=gen[i,j]
            elif np.sum(trans[:,:,i,j] > 0):
                transValue[:,:,i,j]=0
                print("bad", i, j)

    return trans

def sliceTransfer(trans, otherbin, which):
    if which == 'wt':
        if otherbin is not None:
            trans = trans[otherbin,:,otherbin,:]
        else:
            trans = np.sum(trans, axis=(0, 2))
    else:
        if otherbin is not None:
            trans = trans[:, otherbin, :, otherbin]
        else:
            trans = np.nansum(trans, axis=(1, 3))
    return trans

def plotPurityStability(Hdict, ptbin, otherbin, whichbin, includeInefficiency, 
                       purity, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = makeTransfer(Hdict, ptbin, includeInefficiency)
    trans = sliceTransfer(trans, otherbin, whichbin)

    if whichbin == 'dR':
        xs = dRaxis.centers
        edges = dRaxis.edges
        xlabel = "$\Delta R"
    else:
        xs = wtaxis.centers
        edges = wtaxis.edges
        xlabel = "$w"

    if purity:
        ax.set_ylabel("Purity")
        ax.set_xlabel(xlabel+"_{Reco}$")
        values = (np.diag(trans) / np.nansum(trans, axis=1))[1:-1]
    else:
        ax.set_ylabel("Stability")
        ax.set_xlabel(xlabel+"_{Gen}$")
        values = (np.diag(trans) / np.nansum(trans, axis=0))[1:-1]
    
    plotValues(values, 0, xs, label=label, ax=ax)

def showTransfer(Hdict, ptbin, otherbin, which, includeInefficiency, ax=None):
    if ax is None:
        ax = plt.gca()

    if which == 'wt':
        plt.xlabel('Reco weight bin')
        plt.ylabel('Gen weight bin')
    else:
        plt.xlabel("Reco $\Delta R$ bin")
        plt.ylabel("Gen $\Delta R$ bin")

    trans = makeTransfer(Hdict, ptbin, includeInefficiency)
    trans = sliceTransfer(trans, otherbin, which)

    plt.imshow(trans)
    plt.colorbar()
    plt.show()

def transferHist(Hdict, ptbin, thisbin, otherbin, whichbin,
                 includeInefficiency, axis='Gen', 
                 label=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    trans = makeTransfer(Hdict, ptbin, includeInefficiency)
    trans = sliceTransfer(trans, otherbin, whichbin)
    
    if axis == 'Gen':
        values = trans[thisbin, :]/np.sum(trans, axis=1)
        np.nan_to_num(values, copy=False)
        print(np.sum(values))
        xlabel = 'Gen'
    elif axis == 'Reco':
        values = trans[:, thisbin]/np.sum(trans, axis=0)
        np.nan_to_num(values, copy=False)
        print(np.sum(values))
        xlabel = 'Reco'

    if whichbin == 'dR':
        ax.set_xlabel(xlabel+" $\Delta R$ bin")
    else:
        ax.set_xlabel(xlabel+" weight bin")

    ax.hist(np.arange(52), bins=52, weights=values, label=label, histtype='step')
    ax.axvline(thisbin, color='k', linestyle='--')
    if label is not None:
        plt.legend()

def pttitle(title, ptbin, Hdict, fig=None):
    if fig is None:
        fig = plt.gcf()
    if ptbin is None:
        fig.suptitle(title)
    else:
        ptmin = ptaxis.edges[ptbin]
        ptmax = ptaxis.edges[ptbin+1]
        fig.suptitle("%s\n$%0.1f < p_T^{Jet} \\mathrm{[GeV]} < %0.1f$" % (title, ptmin, ptmax))

def plotReco(Hdict, ptbin, folder=None):
    pttitle("Reco EEC", ptbin, Hdict)
    plotEEC(Hdict['Hreco'], label='Total Reco', ptbin=ptbin)
    plotEEC(Hdict['HrecoPUjets'], label='PU Jets', ptbin=ptbin)
    plotEEC(Hdict['HrecoUNMATCH'], label='PU Contamination', ptbin=ptbin)
    if folder is not None:
        plt.savefig("%s/RecoEEC_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def plotGen(Hdict, ptbin, folder=None):
    pttitle("Gen EEC", ptbin, Hdict)
    plotEEC(Hdict['Hgen'], label='Total Gen', ptbin=ptbin)
    plotEEC(Hdict['HgenUNMATCH'], label='Unmatched Gen', ptbin=ptbin)
    if folder is not None:
        plt.savefig("%s/GenEEC_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def compareGenReco(Hdict, ptbin, folder=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, height_ratios=[3,1])
    pttitle("Gen vs Reco EEC", ptbin, Hdict, fig)
    Hreco = Hdict['Hreco'] - Hdict['HrecoPUjets'] - Hdict['HrecoUNMATCH']
    HcovReco = Hdict['HcovReco'] - Hdict['HcovRecoPUjets'] - Hdict['HcovRecoUNMATCH']
    Hgen = Hdict['Hgen'] - Hdict['HgenUNMATCH']
    HcovGen = Hdict['HcovGen'] - Hdict['HcovGenUNMATCH']

    plotEEC(Hreco, label = 'Reco - background', ptbin=ptbin, ax=ax0)
    plotEEC(Hgen, label = 'Gen - unmatched', ptbin=ptbin, ax=ax0)

    plotRatio(Hreco, Hgen, ptbin=ptbin, ax=ax1)
    plt.tight_layout()
    if folder is not None:
        plt.savefig("%s/RecoVsGenEEC_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def comparePurity(Hdicts, labels, ptbin, otherbin, whichbin,
                  includeInefficiency=True, folder=None):
    pttitle("Purity (diagonal pT bins)", ptbin, Hdicts[0])
    for Hdict, label in zip(Hdicts, labels):
        plotPurityStability(Hdict, ptbin, otherbin, whichbin, 
                            label=label, purity=True,
                           includeInefficiency=includeInefficiency)
    if folder is not None:
        plt.savefig("%s/ComparePurity_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def compareStability(Hdicts, labels, ptbin, otherbin, whichbin,
                     includeInefficiency=True, folder=None):
    pttitle("Stability (diagonal pT bins)", ptbin, Hdicts[0])
    for Hdict, label in zip(Hdicts, labels):
        plotPurityStability(Hdict, ptbin, otherbin, whichbin, 
                            label=label, purity=False,
                            includeInefficiency=includeInefficiency)
    if folder is not None:
        plt.savefig("%s/CompareStability_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def compareTransferHist(Hdicts, labels, ptbin, dRbin, includeInefficiency=True, axis='Reco', folder=None):
    pttitle("Slice of transfer matrix", ptbin, Hdicts[0])
    for Hdict, label in zip(Hdicts, labels):
        transferHist(Hdict, ptbin, dRbin, includeInefficiency, axis=axis, label=label)
    plt.yscale('log')
    if folder is not None:
        plt.savefig("%s/CompareTransferHist_%s_ptbin%d.png" % (folder, axis, ptbin), format='png', bbox_inches='tight')
    plt.show()
