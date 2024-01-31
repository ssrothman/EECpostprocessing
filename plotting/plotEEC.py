import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats

import plotting.EECutil

#edges = np.linspace(0, 0.5, 51)
#edges[0] = 1e-10

dRedges = [1e-06, 1e-05, 0.0001, 0.001, 0.003, 
        0.01, 0.015, 0.02, 0.025, 0.03, 
        0.04, 0.05, 0.07, 0.1, 0.15, 
        0.2, 0.3, 0.4, 0.5
]
dRaxis = hist.axis.Variable(dRedges, name='dR', label='$\Delta R$', flow=True)

ptbins = [30, 50, 100, 150, 250, 500]
ptaxis = hist.axis.Variable(ptbins, name='pt')

PUbins = [0, 20, 40, 60, 80]
PUaxis = hist.axis.Variable(PUbins, name='nPU', underflow=False)

def setup_ratiopad(sharex=True):
    return plt.subplots(2, 1, figsize=(5, 6), sharex=sharex, 
                        height_ratios=[3,2])

def reflected_gaussian(binIndex, mu, sigma):
    left = np.where(binIndex!=0, binIndex-0.5, 0)
    right = binIndex+.5

    rightcdf = scipy.stats.norm.cdf(right, loc=mu, scale=sigma)
    leftcdf = scipy.stats.norm.cdf(left, loc=mu, scale=sigma)
    prob_nominal = rightcdf-leftcdf

    mirrorrightcdf = scipy.stats.norm.cdf(-left, loc=mu, scale=sigma)
    mirrorleftcdf = scipy.stats.norm.cdf(-right, loc=mu, scale=sigma)
    prob_mirror = mirrorrightcdf-mirrorleftcdf

    return prob_nominal + prob_mirror

def plotValues(values, errs, xs, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    differences = xs[1:] - xs[:-1]
    differences = np.concatenate(([differences[0]], differences))
    ax.errorbar(xs, values, yerr=errs, xerr = differences/2, ms=3, fmt='o', label=label)
    if label is not None:
        ax.legend()

    plt.tight_layout()
    ax.grid(True)

def applyPlotOptions(values, errs, logwidth, density, dRweight):
    xs = dRaxis.centers
    edges = dRaxis.edges

    if type(density) in [float, int]:
        N = density
        errs = errs/N
        values = values/N
    elif density:
        N = np.sum(values)
        errs = errs*np.sqrt(np.square(1/N) - np.square(values/N**2))
        values = values/N

    values = values[1:-1]
    errs = errs[1:-1]

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

    return values, errs

def plotEEC(EECobj, name, key, 
            ptbin=None, etabin=None, pubin=None,
            logwidth=True, density=False, dRweight=0, 
            label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    vals, errs = EECobj.getProjValsErrs(name, key, ptbin, pubin)
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

def plotForward(EECobj, name, other, othername, 
                ptbin=None, etabin=None, pubin=None,
                logwidth=True, density=False, 
                doTemplates = True,
                label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    vals, errs = EECobj.getForwardValsErrs(name, other, othername, 
                                           ptbin, etabin, pubin,
                                           doTemplates=doTemplates)
    vals, errs = applyPlotOptions(vals, errs, logwidth, density, 0)

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

def plotFactors(EECobj, name, ptbin=None, etabin=None, pubin=None,
                wrt='dR', label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    vals, errs = EECobj.getFactors(name, ptbin, etabin, pubin, wrt)
    vals = vals[1:-1]
    errs = errs[1:-1]

    if wrt=='dR':
        ax.set_xlabel("$\Delta R$")
        xs = dRaxis.centers
    elif wrt=='EECwt':
        ax.set_xlabel("EEC weight")
        ax.set_xscale('log')
        xs = wtaxis.centers
    else:
        raise ValueError("wrt must be 'dR' or 'EECwt'")

    ax.set_ylim(0, 2)
    ax.axhline(1, color='k', linestyle='--')

    plotValues(vals, errs, xs, label=label, ax=ax)

def plotWeights(EECobj, name, key, 
                ptbin=None, etabin=None, pubin=None, dRbin=None, 
                label=None, ax=None):
    raise NotImplementedError

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
    raise NotImplementedError
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

def plotRatio(EECobj1, name1, key1, EECobj2, name2, key2, 
              ptbin1, ptbin2, etabin1, etabin2, pubin1, pubin2,
              logwidth=True, density=False, dRweight=0, 
              mode='ratio', hline=True, label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    val1, err1 = EECobj1.getProjValsErrs(name1, key1, ptbin1, pubin1)
    val1, err1 = applyPlotOptions(val1, err1, logwidth, density, dRweight)

    val2, err2 = EECobj2.getProjValsErrs(name2, key2, ptbin2, pubin2)
    val2, err2 = applyPlotOptions(val2, err2, logwidth, density, dRweight)
    
    if mode is not None:
        return _handleRatio(val1, err1, val2, err2, mode, label, ax, hline=hline)

def plotForwardRatio(transferobj, transfername, dataobj, dataname,
                     ptbin=None, etabin=None, pubin=None,
                     logwidth=True, density=False, dRweight=0,
                     doTemplates = False,
                     mode='ratio', label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    val1, err1 = transferobj.getForwardValsErrs(transfername, 
                                                dataobj, dataname, 
                                                ptbin, etabin, pubin,
                                                doTemplates = doTemplates)
    val1, err1 = applyPlotOptions(val1, err1, logwidth, density, dRweight)

    val2, err2 = dataobj.getProjValsErrs(dataname, 'Hreco' if doTemplates else 'HrecoPure', 
                                     ptbin, pubin)
    val2, err2 = applyPlotOptions(val2, err2, logwidth, density, dRweight)

    if mode is not None:
        _handleRatio(val1, err1, val2, err2, mode, label, ax)

def plotFactorRatio(obj1, name1, obj2, name2, 
                    ptbin1=None, ptbin2=None,
                    etabin1=None, etabin2=None,
                    pubin1=None, pubin2=None,
                    wrt='dR', ax=None):
    if ax is None:
        ax = plt.gca()

    val1,err1 = obj1.getFactors(name1, ptbin1, etabin1, pubin1, wrt)
    val2,err2 = obj2.getFactors(name2, ptbin2, etabin2, pubin2, wrt)

    val1 = val1[1:-1]
    err1 = err1[1:-1]
    val2 = val2[1:-1]
    err2 = err2[1:-1]

    if mode is not None:
        _handleRatio(val1, err1, val2, err2, 'ratio', None, ax)

def _handleRatio(val1, err1, val2, err2,
                 mode, label, ax, hline=True):
    xs = dRaxis.centers
    ax.set_xlabel("$\Delta R$")
    if mode=='difference' or mode=='pulls' or mode=='sigma':
        diff = val1-val2
        differr = np.sqrt(np.square(err1) + np.square(err2))
        if mode=='sigma':
            ax.set_ylabel("Difference [sigma]")
            if hline:
                ax.axhline(0, color='k', linestyle='--')
                ax.fill_between(xs, -1, 1, color='b', alpha=0.2)
            plotValues(diff/differr, 1, xs, label=label, ax=ax)
        elif mode=='difference':
            ax.set_ylabel("Difference")
            if hline:
                ax.axhline(0, color='k', linestyle='--')
            plotValues(diff, differr, xs, label=label, ax=ax)
        else:
            ax.set_ylabel("Pulls")
            ax.hist(diff/differr, histtype='step', bins=np.arange(-6,7)-0.5, label=label, orientation='horizontal')
            if hline:
                ax.axhline(0, color='k', linestyle='--')
            if label is not None:
                ax.legend()
        return diff, differr
    elif mode=='ratio':
        ratio = val1/val2
        ratioerrs = ratio*np.sqrt(np.square(err1/val1) 
                                  + np.square(err2/val2))

        ax.set_ylabel("Ratio")
        if hline:
            ax.axhline(1, color='k', linestyle='--')
        plotValues(ratio, ratioerrs, xs, label=label, ax=ax)
        return ratio, ratioerrs
    else:
        raise ValueError("Mode must be in ['difference', 'pulls', 'ratio']")

def plotPurityStability(EECobj, name, 
                        ptbin, etabin, pubin,
                        purity, otherbin=None, which=None, 
                        label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = EECobj.getDRtransfer(name, etabin, pubin)
    if ptbin is not None:
        trans = trans[ptbin+1,:,ptbin+1,:]
    else:
        trans = np.sum(trans, axis=(0,2))
        

    xs = dRaxis.centers
    xlabel = "$\Delta R"
    
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

def showPtTransfer(EECobj, name, etabin, pubin, ax=None):
    if ax is None:
        ax = plt.gca()

    trans = EECobj.getDRtransfer(name, etabin, pubin)
    trans = np.sum(trans, axis=(1,3))

    plt.imshow(trans, origin='lower', norm=matplotlib.colors.Normalize(1,10))
    plt.xlabel("Reco $p_T$ bin")
    plt.ylabel("Gen $p_T$ bin")
    plt.colorbar()
    plt.show()

def showTransfer(EECobj, name, ptbin, etabin, pubin, 
                 otherbin=None, which=None, ax=None):
    raise NotImplementedError
    if ax is None:
        ax = plt.gca()

    trans = EECobj.getDRtransfer(name)
    if ptbin is not None:
        trans = trans[ptbin+1,:,ptbin+1,:]
    else:
        trans = np.sum(trans, axis=(0,2))
    xtype='dR'

    plt.imshow(trans, origin='lower', norm=matplotlib.colors.LogNorm(1e-5, 1))
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
    raise NotImplementedError
    if ax is None:
        ax = plt.gca()
    
    trans = EECobj.getDRtransfer(name)
    if ptbin is not None:
        trans = trans[ptbin+1, :, ptbin+1, :]
    else:
        trans = np.sum(trans, axis=(0,2))
    trans = np.nan_to_num(trans/np.sum(trans, axis=0))
    
    if axis == 'Gen':
        values = trans[thisbin, :]
        xlabel = 'Gen'
    elif axis == 'Reco':
        values = trans[:, thisbin]
        xlabel = 'Reco'

    values = values/np.sum(values)
    print(np.sum(values))

    xlabel += " $\Delta R$ bin"

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

def plotReco(EECobj, name, ptbin, etabin, pubin, folder=None, logwidth=True):
    pttitle("Reco EEC", ptbin)
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth, density=False)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='Unmatched component',
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth, density=False)
    if folder is not None:
        plt.savefig("%s/RecoEEC_ptbin%d_etabin%d_pubin%d.png" % (folder, ptbin, 
                                                                 etabin, pubin), 
                    format='png', bbox_inches='tight')
    plt.show()

def plotPUShape(EECobj, name, ptbin, etabin, pubin, folder=None, logwidth=True):
    pttitle("Reco EEC Shapes", ptbin)
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='Unmatched component',
            ptbin=ptbin, etabin=etabin, pubin=pubin, 
            logwidth=logwidth, density=True)
    if folder is not None:
        plt.savefig("%s/PUShapes_ptbin%d_etabin%d_pubin%d.png" % (folder, ptbin, 
                                                                  etabin, pubin), 
                    format='png', bbox_inches='tight')
    plt.show()

def plotGen(EECobj, name, ptbin, etabin, pubin, folder=None, logwidth=True):
    pttitle("Gen EEC", ptbin)
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen', 
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth)
    if folder is not None:
        plt.savefig("%s/GenEEC_ptbin%d_etabin%d_pubin%d.png" % (folder, ptbin, 
                                                                etabin, pubin), 
                    format='png', bbox_inches='tight')
    plt.show()


def plotPUjets(EECobj, name, ptbin, etabin, pubin, folder=None):
    fig, (ax0, ax1) = setup_ratiopad()
    pttitle("Pileup jets", ptbin, fig)

    plotEEC(EECobj, name, 'HrecoPUjets', label='PU Jets', 
            ptbin=ptbin, etabin=etabin, pubin=pubin, 
            density=True, ax=ax0)
    plotEEC(EECobj, name, 'HrecoPure', label='Reco Pure',
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            density=True, ax=ax0)

    plotRatio(EECobj, name, 'HrecoPUjets', EECobj, name, 'HrecoPure',
              ptbin1=ptbin, ptbin2=ptbin,
              etabin1=etabin, etabin2=etabin,
              pubin1=pubin, pubin2=pubin,
              density=True, ax=ax1)
    plt.show()

def compareEECratio(EECobjs, names, keys, labels, 
                    ptbins, etabins, pubins,
                    ratio_to, ratio_mode='difference',
                    density=False, folder=None):
    fig, (ax0, ax1) = setup_ratiopad()
    pttitle("Ratios to %s"%ratio_to, None, fig)
    
    vals = []
    errs = []
    for i in range(len(EECobjs)):
        newval, newerr = plotRatio(EECobjs[i], names[i], keys[i], 
                                   EECobjs[i], names[i], ratio_to,
                                   ptbin1=ptbins[i], ptbin2 = ptbins[i],
                                   etabin1=etabins[i], etabin2=etabins[i],
                                   pubin1=pubins[i], pubin2=pubins[i],
                                   density=density, label=labels[i], 
                                   ax=ax0, hline=False)
        vals.append(newval)
        errs.append(newerr)

    if ratio_mode is not None:
        ax1.plot([], [])
        for i in range(1, len(EECobjs)):
            _handleRatio(vals[0], errs[0], vals[i], errs[i], ratio_mode, label=None, ax=ax1)
        ax1.set_ylabel("Ratio to %s"%labels[0])

    if folder is not None:
        plt.savefig("%s/test.png"%folder, format='png', bbox_inches='tight')
    
    plt.show()

def plotUnmatchedShape(EECobj, name, ptbin, etabin, pubin, 
                       folder=None, logwidth=True):
    pttitle("Gen EEC Shapes", ptbin)
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen',
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            logwidth=logwidth, density=True)
    if folder is not None:
        plt.savefig("%s/UnmatchedShapes_ptbin%d_etabin%d_pubin%d.png" % (folder, ptbin,
                                                                         etabin, pubin),
                    format='png', bbox_inches='tight')
    plt.show()

def compareGenReco(EECobj, name, ptbin, etabin, pubin, folder=None):
    fig, (ax0, ax1) = setup_ratiopad()

    pttitle("Gen vs Reco EEC", ptbin, fig)

    plotEEC(EECobj, name, 'HrecoPure', label = 'Reco - background', 
            ptbin=ptbin, etabin=etabin, pubin=pubin, ax=ax0)
    plotEEC(EECobj, name, 'HgenPure', label = 'Gen - unmatched', 
            ptbin=ptbin, etabin=etabin, pubin=pubin, ax=ax0)

    plotRatio(EECobj, name, 'HrecoPure', EECobj, name, 'HgenPure',
              ptbin1=ptbin, ptbin2=ptbin,
              etabin1=etabin, etabin2=etabin,
              pubin1=pubin, pubin2=pubin,
              ax=ax1)

    plt.tight_layout()
    if folder is not None:
        plt.savefig("%s/RecoVsGenEEC_ptbin%d.png" % (folder, ptbin, 
                                                     etabin, pubin), 
                    format='png', bbox_inches='tight')
    plt.show()

def compareReco(EECobj1, name1, label1, 
                EECobj2, name2, label2,
                ptbin1, ptbin2,
                etabin1, etabin2,
                pubin1, pubin2, 
                folder=None):
    fig, (ax0, ax1) = setup_ratiopad()
    pttitle("Reco EEC", ptbin, fig)

    plotEEC(EECobj1, name1, 'Hreco', label = label1, 
            ptbin=ptbin1, etabin=etabin1, pubin=pubin1,
            ax=ax0, density=True)
    plotEEC(EECobj2, name2, 'Hreco', label = label2, 
            ptbin=ptbin2, etabin=etabin2, pubin=pubin2,
            ax=ax0, density=True)

    plotRatio(EECobj1, name1, 'Hreco', EECobj2, name2, 'Hreco', 
              ptbin1=ptbin1, ptbin2=ptbin2,
              etabin1=etabin1, etabin2=etabin2,
              pubin1=pubin1, pubin2=pubin2,
              ax=ax1)
    if folder is not None:
        plt.savefig("%s/CompareReco_ptbin%d_etabin%d_pubin%d.png" %(folder, ptbin,
                                                                    etabin, pubin), 
                    format='png', bbox_inches='tight')
    plt.show()

def comparePurityStability(EECobjs, names, labels, ptbin, etabin, pubin, 
                           purity, otherbin=None, which=None, folder=None):

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
        plotPurityStability(EECobj, name, ptbin, etabin, pubin,
                            label=label, purity=purity,
                            otherbin = otherbin, which=which)
    if folder is not None:
        plt.savefig("%s/Compare%s_ptbin%d_etabin%d_pubin%d_%s.png" % (folder,
                                                                      titlename, ptbin,
                                                                      etabin, 
                                                                      pubin, 
                                                                      which), 
                    format='png', bbox_inches='tight')
    plt.show()

def compareTransferHist(EECobjs, names, labels, 
                        ptbin, etabin, pubin, thisbin, axis='Reco',
                        otherbin=None, which=None, logy=False, folder=None):
    raise NotImplementedError
    if which is None:
        titlestr = "Slice of transfer matrix (diagonal pT bins)"
    elif which == 'dR':
        titlestr='Slice of transfer matrix (diagonal pT bins; integrated over wt bins)'
    elif which == 'wt':
        titlestr='Slice of transfer matrix (diagonal pT bins; integrated over dR bins)'
    pttitle(titlestr, ptbin)

    for EECobj, name, label in zip(EECobjs, names, labels):
        transferHist(EECobj, name, ptbin, etabin, pubin, thisbin, axis=axis, 
                     otherbin=otherbin, which=which, label=label)
    if logy:
        plt.yscale('log')

    if folder is not None:
        plt.savefig("%s/CompareTransferHist_%s_ptbin%d_%s%d.png" % (folder, axis, ptbin, which, thisbin), format='png', bbox_inches='tight')
    plt.show()

def compareGenRecoWeights(EECobj, name, ptbin, dRbin, folder=None):
    raise NotImplementedError
    fix, (ax0, ax1) = setup_ratiopad()
    plotWeights(EECobj, name, 'HrecoPure', ptbin=ptbin, dRbin=dRbin, 
                ax=ax0, label='Reco - background')
    plotWeights(EECobj, name, 'HgenPure', ptbin=ptbin, dRbin=dRbin,
                ax=ax0, label='Gen - unmatched')

    plotWeightRatio(EECobj, name, 'HrecoPure', EECobj, name, 'HgenPure',
                    ptbin=ptbin, dRbin=dRbin, ax=ax1)
    ax1.set_ylim(0.5,1.5)

    plt.show()

def compareEEC(EECobjs, names, keys, labels, ptbins, etabins, pubins, folder=None, ratio_mode='diffrence'):
    fig, (ax0, ax1) = setup_ratiopad()
    pttitle("EEC comparison", None, fig)

    for i in range(len(EECobjs)):
        plotEEC(EECobjs[i], names[i], keys[i], 
                ptbin=ptbins[i], etabin=etabins[i], pubin=pubins[i],
                density=True, label=labels[i], ax=ax0)

    ax1.plot([], [])
    for i in range(1,len(EECobjs)):
        plotRatio(EECobjs[i], names[i], keys[i], EECobjs[0], names[0], keys[0],
                  density=True, mode=ratio_mode, 
                  ptbin1=ptbins[i], ptbin2=ptbins[0],
                  etabin1=etabins[i], etabin2=etabins[0],
                  pubin1=pubins[i], pubin2=pubins[0], ax=ax1)

    ax1.set_ylabel("Ratio to %s"%labels[0])
    if folder is not None:
        raise NotImplementedError

    plt.show()

def compareForward(transferobj, transfername, dataobj, dataname,
                   ptbin=None, etabin=None, pubin=None, 
                   doTemplates = False,
                   mode='ratio', folder=None):
    fig, (ax0, ax1) = setup_ratiopad(mode!='pulls')
    pttitle("Forward transfer test", ptbin, fig)

    plotForward(transferobj, transfername, dataobj, dataname, 
                ptbin=ptbin, etabin=etabin, pubin=pubin,
                doTemplates = doTemplates,
                label = 'Forward', ax=ax0)

    plotEEC(dataobj, dataname, 'Hreco' if doTemplates else 'HrecoPure', 
            ptbin=ptbin, etabin=etabin, pubin=pubin,
            label='Reco-background', ax=ax0)

    plotForwardRatio(transferobj, transfername, dataobj, dataname, 
                     doTemplates = doTemplates,
                     ptbin=ptbin, mode=mode, ax=ax1)

    if folder is not None:
        plt.savefig("%s/test.png"%folder, format='png', bbox_inches='tight')

    plt.show()

def compareForwardPulls(transferobjs, transfernames, dataobjs, datanames,
                        labels, ptbin=None, etabin=None, pubin=None,
                        difference=False, folder=None):
    pttitle("Pulls from forward transfer", ptbin)

    for i in range(len(transferobjs)):
        plotForwardRatio(transferobjs[i], transfernames[i],
                         dataobjs[i], datanames[i],
                         ptbin = ptbin, etabin=etabin,
                         pubin=pubin, mode='pulls',
                         label = labels[i])

    plt.show()

def plotResiduals(EECobjs, names, keys, labels, folder=None):

    val0 = EECobjs[0].Hdict[names[0]][keys[0]].values(flow=True)
    val1 = EECobjs[1].Hdict[names[1]][keys[1]].values(flow=True)

    err0 = EECobjs[0].Hdict[names[0]][EECobjs[0].covnames[keys[0]]].values(flow=True)
    err1 = EECobjs[1].Hdict[names[1]][EECobjs[1].covnames[keys[1]]].values(flow=True)

    diag0 = np.einsum('ijij->ij', err0)
    diag1 = np.einsum('ijij->ij', err1)

    diff = val1 - val0
    differr = err0 + err1
    diagdifferr = np.einsum('ijij->ij', differr)

    mask = ~np.isclose(diagdifferr,0)

    #print(diff[mask])
    #print(val1[mask])
    #print(val0[mask])
    #print(diagdifferr[mask])

    print('val0[0]', np.max(val0[1]))
    print('val1[0]', np.max(val1[1]))
    print('err0[0]', np.max(err0[1]))
    print('err1[0]', np.max(err1[1]))
    print('diag0[0]', np.max(diag0[1]))
    print('diag1[0]', np.max(diag1[1]))
    print('diff[0]', np.max(diff[1]))
    print('differr[0]', np.max(differr[1]))
    print('diagdifferr[0]', np.max(diagdifferr[1]))
     
    print(diag0[1])

    plt.hist(residuals, range=[-3,3], bins=51)
    plt.show()

def compareFactors(EECobjs, names, labels, ptbins, etabins, pubins, 
                   wrt='dR', folder=None):
    fig, (ax0, ax1) = setup_ratiopad()
    pttitle("EEC weight scale factors", None, fig)

    for i in range(len(EECobjs)):
        plotFactors(EECobjs[i], names[i], 
                    ptbin=ptbins[i], etabin=etabins[i], pubin=pubins[i],
                    wrt=wrt, label=labels[i], ax=ax0)

    for i in range(1, len(EECobjs)):
        plotFactorRatio(EECobjs[i], names[i], EECobjs[0], names[0], 
                        ptbin1=ptbins[i], ptbin2=ptbins[0],
                        etabin1=etabins[i], etabin2=etabins[0],
                        pubin1=pubins[i], pubin2=pubins[0],
                        wrt=wrt, ax=ax1)

    ax1.set_ylim(0.5,1.5)

    if folder is not None:
        raise NotImplementedError

    plt.show()
    
