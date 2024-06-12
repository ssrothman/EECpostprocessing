import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats

from plotting.EECstats import applyRelation, diagonal
from plotting.EECutil import EEChistReader
from plotting.util import *

import json

#edges = np.linspace(0, 0.5, 51)
#edges[0] = 1e-10

#dRedges = [1e-3, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 
#           0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

import mplhep as hep

plt.style.use(hep.style.CMS)

def setup_ratiopad(sharex=True):
    return plt.subplots(2, 1, figsize=(10, 10), sharex=sharex, 
                        height_ratios=[3,1])

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

def plotValues(values, errs, xs, xerrs, label=None, ax=None, logx=True):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(xs, values, yerr=errs, xerr = xerrs, 
                fmt='o', label=label)
    if label is not None:
        ax.legend(loc='upper left')

    plt.tight_layout()
    ax.grid(True)
    if logx:
        ax.set_xscale('log')

def applyPlotOptions(values, errs, logwidth, dRweight):
    values = values[1:-1]
    errs = errs[1:-1]

    edges = np.asarray(config.binning.bins.dRedges)
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

def plotEEC(EECobj, name, key, bins={'order' : 0},
            logwidth=True, density=False, dRweight=0, 
            label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    vals, errs = EECobj.getProjValsErrs(name, key, bins.copy(), density)
    vals, errs = applyPlotOptions(vals, errs, logwidth, dRweight)

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

    xs, xerrs = getDRcenters_errs()
    plotValues(vals, errs, xs, xerrs, label=label, ax=ax)

def plotRes3(EECobj, name, key, bins={},
             density=False, logcolor=True,
             ax=None):

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    hep.cms.text("Work in progress", ax=ax, fontsize=22)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax, fontsize = 22)
    vals, errs = EECobj.getRes3ValsErrs(name, key, bins, density)

    vals = vals[1:-1, 1:-1]

    ax.set_ylabel("$\\xi$ bin")
    ax.set_xlabel("$\\phi$ bin")

    tmin = 0
    tmax = np.max(vals)
    tminpos = np.min(vals[vals>0])
    q = ax.imshow(vals, origin='lower', norm=matplotlib.colors.LogNorm(tminpos, tmax) if logcolor else matplotlib.colors.Normalize(tmin, tmax), cmap='Reds')
    plt.gcf().colorbar(q)
    plt.show()

def plotRes3Ratio(obj1, obj2, 
                  name1, name2,
                  key1, key2,
                  bins1, bins2,
                  density, logcolor,
                  ratiomode='ratio',
                  ax=None):
    if ax is None:
        ax = plt.gca()


    hep.cms.text("Work in progress", ax=ax, fontsize=22)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax, fontsize=22)

    vals, errs = obj1.getRelationValsErrs(name1, key1, bins1, density,
                                    obj2, name2, key2, bins2, density,
                                          ratiomode=ratiomode)

    ax.set_ylabel("$\\xi$ bin")
    ax.set_xlabel("$\\phi$ bin")

    vals = vals[1:-1, 1:-1]

    tmin = np.nanmin(vals)
    tmax = np.max(vals[np.isfinite(vals)])
    tminpos = np.nanmin(vals[vals>0])
    print(tmin)
    print(tmax)
    print(tminpos)
    q = ax.imshow(vals, origin='lower', norm=matplotlib.colors.LogNorm(tminpos, tmax) if logcolor else matplotlib.colors.Normalize(tmin, tmax), cmap='Reds')
    plt.gcf().colorbar(q)
    plt.show()


def plotForward(EECobj, name, other, othername, bins={'order' : 0},
                logwidth=True, density=False, 
                doTemplates = True,
                label=None, ax=None):
    if ax is None:
        ax = plt.gca()


    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    vals, errs = EECobj.getForwardValsErrs(name, bins,
                                           other, othername,
                                           doTemplates)
    vals, errs = applyPlotOptions(vals, errs, logwidth, 0)

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

    xs, xerrs = getDRcenters_errs()
    plotValues(vals, errs, xs, xerrs, label=label, ax=ax)

def plotFactors(EECobj, name, bins={'order' : 0},
                label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    F, _ = EECobj.getFactorizedTransfer(name, bins)
    vals = F

    errs = np.zeros_like(F)
    vals = vals[1:-1]
    errs = errs[1:-1]

    ax.set_xlabel("$\Delta R$")
    xs, xerrs = getDRcenters_errs()

    ax.set_ylim(0, 2)
    ax.axhline(1, color='k', linestyle='--')

    ax.set_ylabel("Unfolding 'scale factors'")

    plotValues(vals, errs, xs, xerrs, label=label, ax=ax)

def plotRatio(EECobj1, name1, key1, density1, 
              EECobj2, name2, key2, density2,
              bins1 = {'order' : 0}, bins2 = {'order' : 0},
              logwidth=True, dRweight=0, 
              mode='ratio', hline=True, label=None, 
              ysuffix='', ax=None):
    if ax is None:
        ax = plt.gca()

    if isinstance(EECobj1, EEChistReader):
        vals, errs = EECobj1.getRelationValsErrs(name1, key1, bins1, density1,
                                        EECobj2, name2, key2, bins2, density2,
                                                 ratiomode=mode)
        if mode == 'difference' and key1 != 'factor':
            vals, errs = applyPlotOptions(vals, errs, logwidth, dRweight)
        else:
            vals = vals[1:-1]
            errs = errs[1:-1]
    else:
        vals, covs = applyRelation(EECobj1, np.eye(len(name1))*np.square(name1), 
                                   EECobj2, np.eye(len(name2))*np.square(name2),
                                   None, mode)
        errs = np.sqrt(diagonal(covs))
    
    ax.set_xlabel("$\Delta R$")
    if mode == 'ratio':
        ax.set_ylabel("Ratio"+ysuffix)
        linelevel = 1
    elif mode == 'difference':
        ax.set_ylabel("Difference"+ysuffix)
        linelevel = 0
    elif mode == 'sigma':
        ax.set_ylabel("Difference/Err"+ysuffix)
        linelevel = 0
        ax.fill_between([dRaxis.edges[0], dRaxis.edges[-1]], -1, 1, 
                        color='lightblue', alpha=0.5)

    if hline:
        ax.axhline(linelevel, color='k', linestyle='--')

    xs, xerrs = getDRcenters_errs()

    plotValues(vals, errs, xs, xerrs, label=label, ax=ax)
    return vals, errs

def plotForwardRatio(transferobj, transfername, dataobj, dataname,
                     bins = {'order' : 0},
                     logwidth=True, density=False, dRweight=0,
                     doTemplates = False,
                     mode='ratio', label=None, ax=None):
    raise NotImplementedError
    if ax is None:
        ax = plt.gca()

    val1, err1 = transferobj.getForwardValsErrs(transfername, bins,
                                                dataobj, dataname,
                                                doTemplates)
    val1, err1 = applyPlotOptions(val1, err1, logwidth, density, dRweight)

    val2, err2 = dataobj.getProjValsErrs(dataname, 'Hreco' if doTemplates else 'HrecoPure', bins)
    val2, err2 = applyPlotOptions(val2, err2, logwidth, density, dRweight)

    if mode is not None:
        _handleRatio(val1, err1, val2, err2, mode, label, ax)

def plotFactorRatio(obj1, name1, obj2, name2, 
                    bins1 = {'order' : 0}, 
                    bins2 = {'order' : 0},
                    ax=None,
                    ratio_mode='ratio'):
    if ax is None:
        ax = plt.gca()

    plotRatio(obj1, name1, 'factor', False,
              obj2, name2, 'factor', False,
              bins1, bins2, 
              mode=ratio_mode)

def plotPurityStability(EECobj, name, bins,
                        purity, label=None, ax=None,
                        wrt = 'dRbin', mode='proj',
                        factorized=True):
    if ax is None:
        ax = plt.gca()

    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    if factorized:
        _, trans = EECobj.getFactorizedTransfer(name, bins, keepaxes=[wrt], mode=mode)
    else:
        trans = EECobj.getCombinedTransferMatrix(name, bins, keepaxes=[wrt], mode=mode)

    if wrt == 'dRbin':
        #xs, xerr = getDRcenters_errs()
        xlabel = "$\Delta R"
    elif wrt == 'pt':
        #xs, xerr = getPTcenters_errs()
        xlabel = "Jet $p_T"
    elif wrt == 'btag':
        #xs = [0, 1]
        #xerr = [0, 0]
        xlabel = 'btag$'
        print(trans.shape)
    elif wrt == 'xi':
        xlabel = '$\\xi'
    elif wrt == 'phi':
        xlabel = '$\\phi'
    

    if purity:
        xlabel += '^{Reco}$'
        val = np.diag(trans) / np.sum(trans, axis=1)
        ax.set_ylabel("Purity")
    else:
        xlabel += '^{Gen}$'
        val = np.diag(trans) / np.sum(trans, axis=0)
        ax.set_ylabel("Stability")

    ax.set_xlabel(xlabel)

    ax.set_ylim(0, 1)
    
    xs = np.arange(len(val))
    print(xs.shape)
    xerr = 0.5

    plotValues(val, 0, xs, xerr, label=label, ax=ax, logx=False)

def showTransfer(EECobj, name, bins,
                 ax=None,
                 wrt = 'dRbin', mode='proj',
                 factorized=True, logcolor=False):
    if ax is None:
        ax = plt.gca()

    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    if factorized:
        _, trans = EECobj.getFactorizedTransfer(name, bins, keepaxes=[wrt], mode=mode)
    else:
        trans = EECobj.getCombinedTransferMatrix(name, bins, keepaxes=[wrt], mode=mode)

    tmin = np.min(trans)
    tmax = np.max(trans)
    tminpos = np.min(trans[trans>0])
    plt.imshow(trans, origin='lower', norm=matplotlib.colors.LogNorm(tminpos, tmax) if logcolor else matplotlib.colors.Normalize(tmin, tmax), cmap='Reds')

    plt.xlabel("Reco bin")
    plt.ylabel("Gen bin")

    plt.colorbar()
    plt.sh1ow()

def transferHist(EECobj, name, bins, thebin, 
                 axis='Gen', label=None, ax=None,
                 wrt = 'dRbin', mode='proj',
                 factorized=True):
    if ax is None:
        ax = plt.gca()
    
    hep.cms.text("Work in progress", ax=ax)
    hep.cms.lumitext("$59.53 fb^{-1}$ (13 TeV)", ax=ax)

    if factorized:
        _, trans = EECobj.getFactorizedTransfer(name, bins, keepaxes=[wrt], mode=mode)
    else:
        trans = EECobj.getCombinedTransferMatrix(name, bins, keepaxes=[wrt], mode=mode)
    
    if axis == 'Gen':
        values = trans[thebin, :]
        xlabel = 'Gen'
    elif axis == 'Reco':
        values = trans[:, thebin]
        xlabel = 'Reco'

    values = values/np.sum(values)
    xlabel += " bin"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction transfered from %s bin %d"%('reco' if axis=='Gen' else 'gen', thebin))

    ax.hist(np.arange(len(values)), bins=np.arange(len(values)+1)-0.5, weights=values, label=label, histtype='step')
    ax.axvline(thebin, color='k', linestyle='--')
    if label is not None:
        plt.legend(loc='top left')

def pttitle(title, ptbin, fig=None):
    #TODO
    if fig is None:
        fig = plt.gcf()
    if ptbin is None:
        fig.suptitle(title)
    else:
        ptmin = ptaxis.edges[ptbin]
        ptmax = ptaxis.edges[ptbin+1]
        fig.suptitle("%s\n$%0.1f < p_T^{Jet} \\mathrm{[GeV]} < %0.1f$" % (title, ptmin, ptmax))

def plotReco(EECobj, name, bins = {'order' : 0}, show=True, logwidth=True):
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            bins = bins,
            logwidth=logwidth, density=False)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='Unmatched component',
            bins = bins,
            logwidth=logwidth, density=False)

    if show:
        plt.show()

def plotPUShape(EECobj, name, bins = {'order' : 0}, 
                show=True, logwidth=True):
    plotEEC(EECobj, name, 'Hreco', label='Total Reco', 
            bins = bins,
            logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HrecoUNMATCH', label='Unmatched component',
            bins = bins,
            logwidth=logwidth, density=True)

    if show:
        plt.show()

def plotGen(EECobj, name, bins={'order' : 0}, show=True, logwidth=True):
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            bins = bins,
            logwidth=logwidth)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen', 
            bins = bins,
            logwidth=logwidth)

    if show:
        plt.show()

def compareEECratio_perName(EECobj, names, key, ratio_to,
                            bins, labels,
                            ratio_mode='ratio',
                            density=False, show=True):
    N = len(names)
    compareEECratio([EECobj]*N, names, [key]*N, labels,
                    [bins]*N, 
                    ratio_to, ratio_mode,
                    density, show)

def compareEECratio_perBins(EECobj, name, key, ratio_to,
                          bins_l, labels, 
                          ratio_mode='sigma',
                          density=False, show=True):
    N = len(pubins)
    compareEECratio([EECobj]*N, [name]*N, [key]*N, labels,
                    bins_l,
                    ratio_to, ratio_mode,
                    density, show)

def compareEECratio_perObj(EECobjs, name, key, ratio_to,
                           bins, labels,
                           ratio_mode='sigma',
                           density=False, show=True):
    N = len(EECobjs)
    compareEECratio(EECobjs, [name]*N, [key]*N, labels,
                    [bins]*N,
                    ratio_to, ratio_mode,
                    density, show)

def compareEECratio(EECobjs, names, keys, labels, 
                    bins_l,
                    ratio_to, ratio_mode='difference',
                    density=False, show=True):
    fig, (ax0, ax1) = setup_ratiopad()
    
    net_bins = bins_l[0].items()
    for i in range(1, len(bins_l)):
        net_bins = net_bins & bins_l[i].items()
        
    net_bins = dict(net_bins)

    text = binnedtext(net_bins)
    #ax0.text(0.05, 0.55, text, transform=ax0.transAxes)
    ax0.text(0.55, 0.80, text, transform=ax0.transAxes)

    vals = []
    errs = []
    for i in range(len(EECobjs)):
        newval, newerr = plotRatio(EECobjs[i], names[i], keys[i], density,
                                   EECobjs[i], names[i], ratio_to, density,
                                   bins1 = bins_l[i], bins2=bins_l[i],
                                   label=labels[i], 
                                   ax=ax0, hline=False)
        vals.append(newval)
        errs.append(newerr)

    if ratio_mode is not None:
        if len(EECobjs) > 2:
            ax1.plot([], [])
        for i in range(1, len(EECobjs)):
            plotRatio(vals[i], errs[i], None, None,
                      vals[0], errs[0], None, None,
                      mode=ratio_mode, ysuffix = " to %s"%labels[0])

    ax0.set_ylabel('')

    if show:
        plt.show()

def plotUnmatchedShape(EECobj, name, bins={'order' : 0},
                       show=True, logwidth=True):
    plotEEC(EECobj, name, 'Hgen', label='Total Gen',
            bins = bins,
            logwidth=logwidth, density=True)
    plotEEC(EECobj, name, 'HgenUNMATCH', label='Unmatched Gen',
            bins = bins,
            logwidth=logwidth, density=True)

    if show:
        plt.show()

def compareGenReco(EECobj, name, bins = {'order' : 0}, show=True):
    fig, (ax0, ax1) = setup_ratiopad()

    plotEEC(EECobj, name, 'HrecoPure', label = 'Reco - background', 
            bins=bins, ax=ax0)
    plotEEC(EECobj, name, 'HgenPure', label = 'Gen - unmatched', 
            bins=bins, ax=ax0)

    plotRatio(EECobj, name, 'HrecoPure', False,
              EECobj, name, 'HgenPure', False,
              bins1=bins, bins2=bins,
              ax=ax1)

    plt.tight_layout()

    if show:
        plt.show()

def comparePurityStability_perObj(EECobjs, name, labels, bins,
                                  purity, show=True):
    N = len(labels)
    comparePurityStability(EECobjs, [name]*N, labels, [bins]*N,
                           purity, show)

def comparePurityStability(EECobjs, names, labels, bins_l, 
                           purity, show=True):

    if purity:
        titlename = "Purity"
    else:
        titlename = "Stability"

    titlestr = '%s (diagonal pT bins)'%titlename

    for EECobj, name, label, bins in zip(EECobjs, names, labels, bins_l):
        plotPurityStability(EECobj, name, bins=bins,
                            label=label, purity=purity)
    if show:
        plt.show()

def compareTransferHist(EECobjs, names, labels, 
                        bins_l, dRbins, axis='Reco',
                        logy=False, show=True):
    titlestr = "Slice of transfer matrix (diagonal pT bins)"

    for EECobj, name, label, bins, dRbin in zip(EECobjs, names, 
                                                labels, bins_l,
                                                dRbins):
        transferHist(EECobj, name, bins, dRbin = dRbin, axis=axis, label=label)
    if logy:
        plt.yscale('log')

    if show:
        plt.show()

def compareEEC_perBins(EECobj, name, key, labels,
                     bins_l, density,
                     show=True, ratio_mode='ratio'):
    N = len(labels)
    compareEEC([EECobj]*N, [name]*N, [key]*N, labels,
               bins_l, [density]*N,
               show=show, ratio_mode=ratio_mode)

def compareEEC_perObj(EECobjs, name, key, labels,
                      bins, density,
                      show=True, ratio_mode='ratio'):
    N = len(labels)
    compareEEC(EECobjs, [name]*N, [key]*N, labels,
               [bins]*N, [density]*N,
               show=show, ratio_mode=ratio_mode)

def compareEEC(EECobjs, names, keys, labels, bins_l, densities,
               show=True, ratio_mode='ratio'):
    fig, (ax0, ax1) = setup_ratiopad()

    net_bins = bins_l[0].items()
    for i in range(1, len(bins_l)):
        net_bins = net_bins & bins_l[i].items()
        
    net_bins = dict(net_bins)

    text = binnedtext(net_bins)
    ax0.text(0.05, 0.55, text, transform=ax0.transAxes)

    for i in range(len(EECobjs)):
        plotEEC(EECobjs[i], names[i], keys[i], 
                bins=bins_l[i],
                density=densities[i], label=labels[i], ax=ax0)

    if len(EECobjs)>2:
        ax1.plot([], [])
    for i in range(1,len(EECobjs)):
        plotRatio(EECobjs[i], names[i], keys[i], densities[i],
                  EECobjs[0], names[0], keys[0], densities[0],
                  mode=ratio_mode, 
                  bins1=bins_l[i], bins2=bins_l[0], ax=ax1,
                  ysuffix = " to %s"%labels[0])

    if show:
        plt.show()

def compareForward(transferobj, transfername, dataobj, dataname,
                   bins={'order' : 0},
                   doTemplates = False,
                   mode='ratio', show=True,
                   density=False):
    raise NotImplementedError
    fig, (ax0, ax1) = setup_ratiopad(mode!='pulls')

    plotForward(transferobj, transfername, dataobj, dataname, 
                bins=bins,
                doTemplates = doTemplates,
                label = 'Forward', ax=ax0,
                density=density)

    plotEEC(dataobj, dataname, 'Hreco' if doTemplates else 'HrecoPure', 
            bins=bins,
            label='Reco' if doTemplates else 'Reco-background', ax=ax0, 
            density=density)

    plotForwardRatio(transferobj, transfername, dataobj, dataname, 
                     doTemplates = doTemplates,
                     bins=bins, mode=mode, ax=ax1,
                     density=density)

    if show:
        plt.show()

def compareForwardPulls(transferobjs, transfernames, dataobjs, datanames,
                        labels,bins={'order' : 0},
                        difference=False, show=True):

    for i in range(len(transferobjs)):
        plotForwardRatio(transferobjs[i], transfernames[i],
                         dataobjs[i], datanames[i],
                         bins=bins, mode='pulls',
                         label = labels[i])

    plt.show()

def compareFactors_perObj(EECobjs, name, labels, bins, show=True):
    N = len(labels)
    compareFactors(EECobjs, [name]*N, labels, [bins]*N, show)

def compareFactors(EECobjs, names, labels, bins_l,
                   show=True):
    fig, (ax0, ax1) = setup_ratiopad()

    for i in range(len(EECobjs)):
        plotFactors(EECobjs[i], names[i], 
                    bins = bins_l[i],
                    label=labels[i], ax=ax0)

    for i in range(1, len(EECobjs)):
        plotFactorRatio(EECobjs[i], names[i], EECobjs[0], names[0], 
                        bins1=bins_l[i], bins2=bins_l[0],
                        ax=ax1)

    ax1.set_ylim(0.5,1.5)

    if show:
        plt.show()
