import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import plotting.EECutil

edges = np.linspace(0, 0.5, 51)
edges[0] = 1e-10
dRaxis = hist.axis.Variable(edges, name='dR', label='$\Delta R$')
ptaxis = hist.axis.Regular(10, 0, 500)
wtaxis = hist.axis.Regular(25, 1e-6, 1, transform=hist.axis.transform.log)

def plotValues(values, errs, xs, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    differences = xs[1:] - xs[:-1]
    differences = np.concatenate(([differences[0]], differences))
    ax.errorbar(xs, values, yerr=errs, xerr = differences/2, ms=3, fmt='o', label=label)
    if label is not None:
        ax.legend()

def applyPlotOptions(values, errs, logwidth, density, dRweight):
    xs = dRaxis.centers
    edges = dRaxis.edges

    if logwidth:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    values = values/widths
    errs = errs/widths

    if dRweight != 0:
        wt = np.power(xs, dRweight)
        values = values*wt
        errs = errs*wt

    if density:
        N = np.sum(values)
        values = values/N
        errs = errs/N

    return values, errs

def plotEEC(EECobj, name, key, ptbin=None, 
            logwidth=True, density=False, dRweight=0, 
            label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    vals, errs = EECobj.getValsErrs(name, key, ptbin)
    vals = vals[1:-1]
    errs = errs[1:-1]
    vals, errs = applyPlotOptions(vals, errs, logwidth, density, dRweight)

    ax.set_xlabel("$\Delta R$")
    if logwidth:
        ylabel="$\\frac{d\\sigma^{(2)}}{d\\log\\Delta R}$"
    else:
        ylabel="$\\frac{d\\sigma^{(2)}}{d\\Delta R}$"

    if density:
        ylabel += ' [Density]'
    else:
        ylabel += ' [Unnormalized]'

    ax.set_ylabel(ylabel)

    xs = dRaxis.centers
    plotValues(vals, errs, xs, label=label, ax=ax)

def plotWeights(EECobj, name, key, ptbin=None, dRbin=None, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    wts = EECobj.getWeights(name, key, ptbin, dRbin)[1:-1]

    ax.set_xlabel("$wt$")
    ax.set_ylabel("Counts [A.U.]")
    
    titlestr = "EEC weights distribution"
    if dRbin is not None:
        titlestr += " for $\Delta R$ bin {}".format(dRbin)
    pttitle(titlestr, ptbin)

    xs = wtaxis.centers
    ax.set_xscale('log')
    print(np.sum(xs*wts))
    plotValues(wts, np.zeros_like(wts), xs, label=label, ax=ax)

def plotWeightRatio(EECobj1, name1, key1, EECobj2, name2, key2, 
                    ptbin=None, dRbin=None, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    val1 = EECobj1.getWeights(name1, key1, ptbin, dRbin)[1:-1]
    val2 = EECobj2.getWeights(name2, key2, ptbin, dRbin)[1:-1]
    
    ratio = val1/val2
    
    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylabel("Ratio")
    ax.set_xlabel("$wt$")
    xs = wtaxis.centers
    ax.set_xscale('log')
    plotValues(ratio, np.zeros_like(ratio), xs, label=label, ax=ax)

def plotRatio(EECobj1, name1, key1, EECobj2, name2, key2, ptbin=None, 
              logwidth=True, density=False, dRweight=0, kind='EEC',
              label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    val1, err1 = EECobj1.getValsErrs(name1, key1, ptbin)
    val1 = val1[1:-1]
    err1 = err1[1:-1]
    val1, err1 = applyPlotOptions(val1, err1, logwidth, density, dRweight)

    val2, err2 = EECobj2.getValsErrs(name2, key2, ptbin)
    val2 = val2[1:-1]
    err2 = err2[1:-1]
    val2, err2 = applyPlotOptions(val2, err2, logwidth, density, dRweight)

    ratio = val1/val2
    ratioerrs = ratio*np.sqrt(np.square(err1/val1) 
                              + np.square(err2/val2))

    ax.axhline(1, color='k', linestyle='--')
    ax.set_ylabel("Ratio")
    ax.set_xlabel("$\Delta R$")
    xs = dRaxis.centers
    plotValues(ratio, ratioerrs, xs, label=label, ax=ax)

def plotPurityStability(EECobj, name, ptbin, purity,
                        otherbin=None, which=None, 
                        label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if type(EECobj) is plotting.EECutil.EEC:
        trans = EECobj.getTransfer(name, ptbin)
        xtype='dR'
    else:
        trans = EECobj.getSlicedTransfer(name, ptbin, otherbin, which)
        xtype = 'dR' if which == 'dR' else 'wt'

    if xtype == 'dR':
        xs = dRaxis.centers
        xlabel = "$\Delta R"
    else:
        xs = wtaxis.centers
        xlabel = "$wt"
        ax.set_xscale('log')
    
    if purity:
        xlabel += '_{Reco}$'
        val = np.diag(trans) / np.sum(trans, axis=1)
    else:
        xlabel += '_{Gen}$'
        val = np.diag(trans) / np.sum(trans, axis=0)

    ax.set_ylabel("Purity")
    ax.set_xlabel(xlabel)

    ax.set_ylim(0, 1)
    
    plotValues(val[1:-1], 0, xs, label=label, ax=ax)

def showPtTransfer(EECobj, name, ax=None):
    if ax is None:
        ax = plt.gca()

    if type(EECobj) is plotting.EECutil.EEC:
        trans = EECobj.getRawTransfer(name)
        trans = np.sum(trans, axis=(1,3))
    else:
        trans = EECobj.getRawTransfer(name)
        trans = np.sum(trans, axis=(1,2,4,5))

    plt.imshow(trans)
    plt.xlabel("Reco $p_T$ bin")
    plt.ylabel("Gen $p_T$ bin")
    plt.colorbar()
    plt.show()

def showTransfer(EECobj, name, ptbin, otherbin=None, which=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if type(EECobj) is plotting.EECutil.EEC:
        trans = EECobj.getTransfer(name, ptbin)
        xtype='dR'
    else:
        trans = EECobj.getSlicedTransfer(name, ptbin, otherbin, which)
        xtype = 'dR' if which == 'dR' else 'wt'

    plt.imshow(trans)
    if xtype == 'dR':
        plt.xlabel("Reco $\Delta R$ bin")
        plt.ylabel("Gen $\Delta R$ bin")
    else:
        plt.xlabel("Reco $wt$ bin")
        plt.ylabel("Gen $wt$ bin")
    plt.colorbar()
    plt.show()

def transferHist(EECobj, name, ptbin, thisbin, axis='Gen',  
                 otherbin=None, which=None, 
                 label=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    if type(EECobj) is plotting.EECutil.EEC:
        trans = EECobj.getTransfer(name, ptbin)
        xtype='dR'
    else:
        trans = EECobj.getSlicedTransfer(name, ptbin, otherbin, which)
        xtype = 'dR' if which == 'dR' else 'wt'
    
    if axis == 'Gen':
        values = trans[thisbin, :]
        xlabel = 'Gen'
    elif axis == 'Reco':
        values = trans[:, thisbin]
        xlabel = 'Reco'

    values = values/np.sum(values)
    print(np.sum(values))

    if xtype == 'dR':
        xlabel += " $\Delta R$ bin"
    else:
        xlabel += " $wt$ bin"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction transfered from %s bin %d"%('reco' if axis=='Gen' else 'gen', thisbin))

    ax.hist(np.arange(len(values)), bins=np.arange(len(values)+1)-0.5, weights=values, label=label, histtype='step')
    ax.axvline(thisbin, color='k', linestyle='--')
    if label is not None:
        plt.legend()

def pttitle(title, ptbin, fig=None):
    if fig is None:
        fig = plt.gcf()
    if ptbin is None:
        fig.suptitle(title)
    else:
        ptmin = ptaxis.edges[ptbin]
        ptmax = ptaxis.edges[ptbin+1]
        fig.suptitle("%s\n$%0.1f < p_T^{Jet} \\mathrm{[GeV]} < %0.1f$" % (title, ptmin, ptmax))

def plotReco(EECobj, name, ptbin, folder=None, logwidth=True):
    pttitle("Reco EEC", ptbin)
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            ptbin=ptbin, logwidth=logwidth, density=False)
    plotEEC(EECobj, name, 'HrecoPUjets', label='PU Jets', 
            ptbin=ptbin, logwidth=logwidth, density=False)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='PU Contamination',
            ptbin=ptbin, logwidth=logwidth, density=False)
    if folder is not None:
        plt.savefig("%s/RecoEEC_ptbin%d.png" % (folder, ptbin), 
                    format='png', bbox_inches='tight')
    plt.show()

def plotPUShape(EECobj, name, ptbin, folder=None, logwidth=True):
    pttitle("Reco EEC Shapes", ptbin)
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            ptbin=ptbin, logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HrecoPUjets', label='PU Jets', 
            ptbin=ptbin, logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='PU Contamination',
            ptbin=ptbin, logwidth=logwidth, density=True)
    if folder is not None:
        plt.savefig("%s/PUShapes_ptbin%d.png" % (folder, ptbin), 
                    format='png', bbox_inches='tight')
    plt.show()

def plotGen(EECobj, name, ptbin, folder=None, logwidth=True):
    pttitle("Gen EEC", ptbin)
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            ptbin=ptbin, logwidth=logwidth)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen', 
            ptbin=ptbin, logwidth=logwidth)
    if folder is not None:
        plt.savefig("%s/GenEEC_ptbin%d.png" % (folder, ptbin), 
                    format='png', bbox_inches='tight')
    plt.show()

def plotUnmatchedShape(EECobj, name, ptbin, folder=None, logwidth=True):
    pttitle("Gen EEC Shapes", ptbin)
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            ptbin=ptbin, logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen',
            ptbin=ptbin, logwidth=logwidth, density=True)
    if folder is not None:
        plt.savefig("%s/UnmatchedShapes_ptbin%d.png" % (folder, ptbin), 
                    format='png', bbox_inches='tight')
    plt.show()

def compareGenReco(EECobj, name, ptbin, folder=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, 
                                   height_ratios=[3,1])
    pttitle("Gen vs Reco EEC", ptbin, fig)

    plotEEC(EECobj, name, 'HrecoPure', label = 'Reco - background', ptbin=ptbin, ax=ax0)
    plotEEC(EECobj, name, 'HgenPure', label = 'Gen - unmatched', ptbin=ptbin, ax=ax0)

    plotRatio(EECobj, name, 'HrecoPure', EECobj, name, 'HgenPure',
              ptbin=ptbin, ax=ax1)

    plt.tight_layout()
    if folder is not None:
        plt.savefig("%s/RecoVsGenEEC_ptbin%d.png" % (folder, ptbin), format='png', 
                    bbox_inches='tight')
    plt.show()

def testForward(EECobj, name, ptbin, folder=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5), sharex=True,
                                   height_ratios=[3,1])
    pttitle("Forward transfer EEC", ptbin, fig)

    plotEEC(EECobj, name, 'forward', label='Foward transfered', ptbin=ptbin, ax=ax0)
    plotEEC(EECobj, name, 'HrecoPure', label='Reco - background', ptbin=ptbin, ax=ax0)

    plotRatio(EECobj, name, 'forward', EECobj, name, 'HrecoPure',
              ptbin=ptbin, ax=ax1)
    
    ax1.set_ylim(0.9, 1.1)

    plt.tight_layout()
    if folder is not None:
        plt.savefig('%s/TestTransfer_ptbin%d.png'%(folder, ptbin), format='png',
                    bbox_inches='tight')

    plt.show()

def compareReco(EECobj1, name1, label1, 
                EECobj2, name2, label2,
                ptbin, folder=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5), sharex=True,
                                   height_ratios=[3,1])
    pttitle("Reco EEC", ptbin, fig)

    plotEEC(EECobj1, name1, 'Hreco', label = label1, ptbin=ptbin, ax=ax0, density=True)
    plotEEC(EECobj2, name2, 'Hreco', label = label2, ptbin=ptbin, ax=ax0, density=True)

    plotRatio(EECobj1, name1, 'Hreco', EECobj2, name2, 'Hreco', 
              ptbin=ptbin, ax=ax1)
    if folder is not None:
        plt.savefig("%s/CompareReco_ptbin%d.png" %(folder, ptbin), format='png',
                    bbox_inches='tight')
    plt.show()

def comparePurityStability(EECobjs, names, labels, ptbin, purity,
                          otherbin=None, which=None, folder=None):

    if purity:
        titlename = "Purity"
    else:
        titlename = "Stability"

    if which is None:
        titlestr = "%s (diagonal pT bins)"%titlename
    elif which == 'dR':
        titlestr = '%s (diagonal pT bins; integrated over wt bins)'%titlename
    elif which == 'wt':
        titlestr = '%s (diagonal pT bins; integrated over dR bins)'%titlename
    pttitle(titlestr, ptbin)

    for EECobj, name, label in zip(EECobjs, names, labels):
        plotPurityStability(EECobj, name, ptbin, label=label, purity=purity,
                            otherbin = otherbin, which=which)
    if folder is not None:
        plt.savefig("%s/Compare%s_ptbin%d_%s.png" % (folder, titlename, ptbin, which), format='png', bbox_inches='tight')
    plt.show()

def compareTransferHist(EECobjs, names, labels, ptbin, thisbin, axis='Reco',
                        otherbin=None, which=None, logy=False, folder=None):
    if which is None:
        titlestr = "Slice of transfer matrix (diagonal pT bins)"
    elif which == 'dR':
        titlestr='Slice of transfer matrix (diagonal pT bins; integrated over wt bins)'
    elif which == 'wt':
        titlestr='Slice of transfer matrix (diagonal pT bins; integrated over dR bins)'
    pttitle(titlestr, ptbin)

    for EECobj, name, label in zip(EECobjs, names, labels):
        transferHist(EECobj, name, ptbin, thisbin, axis=axis, 
                     otherbin=otherbin, which=which, label=label)
    if logy:
        plt.yscale('log')

    if folder is not None:
        plt.savefig("%s/CompareTransferHist_%s_ptbin%d_%s%d.png" % (folder, axis, ptbin, which, thisbin), format='png', bbox_inches='tight')
    plt.show()

def compareGenRecoWeights(EECobj, name, ptbin, dRbin, folder=None):
    fix, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5), sharex=True,
                                   height_ratios=[3,1])
    plotWeights(EECobj, name, 'HrecoPure', ptbin=ptbin, dRbin=dRbin, 
                ax=ax0, label='Reco - background')
    plotWeights(EECobj, name, 'HgenPure', ptbin=ptbin, dRbin=dRbin,
                ax=ax0, label='Gen - unmatched')

    plotWeightRatio(EECobj, name, 'HrecoPure', EECobj, name, 'HgenPure',
                    ptbin=ptbin, dRbin=dRbin, ax=ax1)
    ax1.set_ylim(0.5,1.5)

    plt.show()
