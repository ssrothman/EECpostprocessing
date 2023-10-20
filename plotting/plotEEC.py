import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import util.EECutil

edges = np.linspace(0, 0.5, 51)
edges[0] = 1e-10
dRaxis = hist.axis.Variable(edges, name='dR', label='$\Delta R$')

def plotValues(values, errs, xs, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    differences = xs[1:] - xs[:-1]
    differences = np.concatenate(([differences[0]], differences))
    ax.errorbar(xs, values, yerr=errs, xerr = differences/2, ms=3, fmt='o', label=label)
    if label is not None:
        ax.legend()

def applyPtBin(Hval, Hcov, ptbin):
    if ptbin is not None:
        Hval = Hval[{'pt':ptbin}]
        Hcov = Hcov[{'pt1':ptbin, 'pt2':ptbin}]

    Hval = Hval.project('dRbin')
    Hcov = Hcov.project('dRbin1', 'dRbin2')

    return Hval, Hcov


def plotEEC(Hval, Hcov, ptbin=None, label=None, logwidth=True, ax=None):
    if ax is None:
        ax = plt.gca()

    Hval, Hcov = applyPtBin(Hval, Hcov, ptbin)

    values = Hval.values(flow=False)[1:-1]
    variances = np.diag(Hcov.values(flow=False)[1:-1,1:-1])
    errs = np.sqrt(variances)

    xs = dRaxis.centers
    edges = dRaxis.edges
    if logwidth:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    values = values/widths
    errs = errs/widths

    ax.set_xlabel("$\Delta R$")
    if logwidth:
        ax.set_ylabel("$\\frac{d\\sigma^{(2)}}{d\\log\\Delta R}$ [Unnormalized]")
    else:
        ax.set_ylabel("$\\frac{d\\sigma^{(2)}}{d\\Delta R}$ [Unnormalized]")

    plotValues(values, errs, xs, label=label, ax=ax)

def plotRatio(Hval1, Hcov1, Hval2, Hcov2, ptbin=None, logwidth=True, 
              label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    Hval1, Hcov1 = applyPtBin(Hval1, Hcov1, ptbin)
    Hval2, Hcov2 = applyPtBin(Hval2, Hcov2, ptbin)

    values1 = Hval1.values(flow=False)[1:-1]
    variances1 = np.diag(Hcov1.values(flow=False)[1:-1,1:-1])
    errs1 = np.sqrt(variances1)

    values2 = Hval2.values(flow=False)[1:-1]
    variances2 = np.diag(Hcov2.values(flow=False)[1:-1,1:-1])
    errs2 = np.sqrt(variances2)

    xs = dRaxis.centers
    edges = dRaxis.edges
    if logwidth:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    values1 = values1/widths
    errs1 = errs1/widths

    values2 = values2/widths
    errs2 = errs2/widths

    ratio = values1/values2
    ratioerrs = ratio*np.sqrt(np.square(errs1/values1) 
                              + np.square(errs2/values2))

    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylabel("Ratio")
    ax.set_xlabel("$\Delta R$")
    plotValues(ratio, ratioerrs, xs, label=label, ax=ax)

def makeTransfer(Hdict, ptbin, includeInefficiency=True):
    Hgen = Hdict['Hgen']
    HcovGen = Hdict['HcovGen']

    if not includeInefficiency:
        Hgen = Hgen - Hdict['HgenUNMATCH']
        HcovGen = HcovGen - Hdict['HcovGenUNMATCH']

    Hgen, HcovGen = applyPtBin(Hgen, HcovGen, ptbin)

    Htrans = Hdict['Htrans']
    Htrans = Htrans[{'ptReco' : ptbin, 'ptGen' : ptbin}].project('dRbinReco', 'dRbinGen')

    transValue = Htrans.values(flow=False)
    genValue = Hgen.values(flow=False)

    target = np.sum(transValue, axis=1)

    transValue = transValue / genValue[None, :]

    return transValue

def plotPurity(Hdict, ptbin, includeInefficiency, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = makeTransfer(Hdict, ptbin, includeInefficiency)

    xs = dRaxis.centers
    edges = dRaxis.edges

    ax.set_xlabel("$\Delta R_{Reco}$")
    ax.set_ylabel("Purity")

    purity = np.diag(trans) / np.sum(trans, axis=1)

    purity = purity[1:-1]
    
    plotValues(purity, 0, xs, label=label, ax=ax)

def plotStability(Hdict, ptbin, includeInefficiency, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = makeTransfer(Hdict, ptbin, includeInefficiency)

    xs = dRaxis.centers
    edges = dRaxis.edges

    ax.set_xlabel("$\Delta R_{Gen}$")
    ax.set_ylabel("Stability")

    purity = np.diag(trans) / np.sum(trans, axis=0)

    purity = purity[1:-1]
    
    plotValues(purity, 0, xs, label=label, ax=ax)

def showTransfer(Hdict, ptbin, includeInefficiency, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = makeTransfer(Hdict, ptbin, includeInefficiency)
    plt.imshow(trans)
    plt.show()

def transferHist(Hdict, ptbin, dRbin, includeInefficiency, axis='Gen', 
                 label=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    trans = makeTransfer(Hdict, ptbin, includeInefficiency)
    
    if axis == 'Gen':
        values = trans[dRbin, :]/np.sum(trans, axis=1)
        print(np.sum(values))
        ax.set_xlabel('Gen $\Delta R$ bin')
    elif axis == 'Reco':
        values = trans[:, dRbin]/np.sum(trans, axis=0)
        print(np.sum(values))
        ax.set_xlabel('Reco $\Delta R$ bin')

    ax.hist(np.arange(52), bins=52, weights=values, label=label, histtype='step')
    ax.axvline(dRbin, color='k', linestyle='--')
    if label is not None:
        plt.legend()

def pttitle(title, ptbin, Hdict, fig=None):
    if fig is None:
        fig = plt.gcf()
    if ptbin is None:
        fig.suptitle(title)
    else:
        ptmin = Hdict['Hreco'].axes['pt'].edges[ptbin]
        ptmax = Hdict['Hreco'].axes['pt'].edges[ptbin+1]
        fig.suptitle("%s\n$%0.1f < p_T^{Jet} \\mathrm{[GeV]} < %0.1f$" % (title, ptmin, ptmax))

def plotReco(Hdict, ptbin, folder=None):
    pttitle("Reco EEC", ptbin, Hdict)
    plotEEC(Hdict['Hreco'], Hdict['HcovReco'], label='Total Reco', ptbin=ptbin)
    plotEEC(Hdict['HrecoPUjets'], Hdict['HcovRecoPUjets'], label='PU Jets', ptbin=ptbin)
    plotEEC(Hdict['HrecoUNMATCH'], Hdict['HcovRecoUNMATCH'], label='PU Contamination', ptbin=ptbin)
    if folder is not None:
        plt.savefig("%s/RecoEEC_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def plotGen(Hdict, ptbin, folder=None):
    pttitle("Gen EEC", ptbin, Hdict)
    plotEEC(Hdict['Hgen'], Hdict['HcovGen'], label='Total Gen', ptbin=ptbin)
    plotEEC(Hdict['HgenUNMATCH'], Hdict['HcovGenUNMATCH'], label='Unmatched Gen', ptbin=ptbin)
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

    plotEEC(Hreco, HcovReco, label = 'Reco - background', ptbin=ptbin, ax=ax0)
    plotEEC(Hgen, HcovGen, label = 'Gen - unmatched', ptbin=ptbin, ax=ax0)

    plotRatio(Hreco, HcovReco, Hgen, HcovGen, ptbin=ptbin, ax=ax1)
    plt.tight_layout()
    if folder is not None:
        plt.savefig("%s/RecoVsGenEEC_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def comparePurity(Hdicts, labels, ptbin, includeInefficiency=True, folder=None):
    pttitle("Purity (diagonal pT bins)", ptbin, Hdicts[0])
    for Hdict, label in zip(Hdicts, labels):
        plotPurity(Hdict, ptbin, label=label,
                   includeInefficiency=includeInefficiency)
    if folder is not None:
        plt.savefig("%s/ComparePurity_ptbin%d.png" % (folder, ptbin), format='png', bbox_inches='tight')
    plt.show()

def compareStability(Hdicts, labels, ptbin, includeInefficiency=True, folder=None):
    pttitle("Stability (diagonal pT bins)", ptbin, Hdicts[0])
    for Hdict, label in zip(Hdicts, labels):
        plotStability(Hdict, ptbin, label=label, 
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
