import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import EECutil

def plotvalues(vals, errs, axis, label=None, dlog=True, savefig=None, clear=False, show=True, ax=None, logy=True, xlab=True):
    if ax is None:
        ax = plt.gca()

    centers = axis.centers
    edges = axis.edges

    if dlog:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    vals=vals/widths
    errs=errs/widths

    ax.set_xscale('log')

    ax.errorbar(centers, vals, yerr=errs, fmt='o--', label=label)
    if xlab:
        ax.set_xlabel("$\Delta R_{max}$", fontsize=20)

    if label is not None:
        ax.legend()

    if logy:
        ax.set_yscale('log')

    if savefig is not None:
        plt.tight_layout()
        low, high=ax.get_ylim()
        ax.set_ylim(1e-4, high)
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
    if show:
        plt.tight_layout()
        low, high=ax.get_ylim()
        ax.set_ylim(1e-4, high)
        plt.show()
    if clear:
        plt.clf()

def addYlabel(order, dlog=True, ax=None):
    if ax is None:
        ax = plt.gca()
    if dlog:
        ax.set_ylabel("$\\frac{d\\sigma^{[%d]}}{d\\log\\Delta R_{max}}$"%order, fontsize=20)
    else:
        ax.set_ylabel("$\\frac{d\\sigma^{[%d]}}{d\\Delta R_{max}}$"%order, fontsize=20)

def plotForward(htrans, hreco, hgen, hcovgen, order,
                label=None, dlog=True, savefig=None, clear=False, show=True,
                ax=None):
    genvals, gencov = EECutil.getProjectedValsCov(hgen, hcovgen, order=order, 
                                                  flow=True, normalize=False)

    transMat = EECutil.getTransferMatrix(htrans, hgen, hreco, order)

    forward, covforward = EECutil.forwardTransfer(transMat, genvals, gencov)

    vals = forward[1:-1]
    errs = np.sqrt(np.diag(covforward))[1:-1]

    addYlabel(order, dlog, ax)
    plotvalues(vals, errs, hreco.axes['dR'], label, dlog, savefig, clear, show, ax)

def plotForwardCross(h1trans, h1reco, h1covreco, h1gen, h1covgen, 
                     h2reco, h2covreco, h2gen, h2covgen, order,
                     dlog=True, savefig=None, clear=False, show=True,
                     ax=None):

    transMat = EECutil.getTransferMatrix(h1trans, h1gen, h1reco, order=order)

    gen2vals, gen2cov = EECutil.getProjectedValsCov(h2gen, h2covgen, order=order,
                                                    flow=True, normalize=False)
    forward, covforward = EECutil.forwardTransfer(transMat, gen2vals, gen2cov)

    forwardvals = forward[1:-1]
    forwarderrs = np.sqrt(np.diag(covforward))[1:-1]
    print("FORWARD[15] = ", forwardvals[15])

    addYlabel(order, dlog, ax)
    plotvalues(forwardvals, forwarderrs, h2reco.axes['dR'], "Forward", dlog, savefig=None, clear=False, show=False, ax=ax)
    
    template, templatecov = EECutil.getBackgroundTemplate(h1trans, 
                                                          h1gen, h1covgen,
                                                          h1reco, h1covreco,
                                                          order = order,
                                                          normByGen=True)
    templnorm = gen2vals.sum()
    templvals = template[1:-1]*templnorm
    templerrs = np.sqrt(np.diag(templatecov))[1:-1]*templnorm
    print("BACKGROUND[15] = ", templvals[15])
    
    plotvalues(templvals, templerrs, h2reco.axes['dR'], "Background", dlog, savefig=None, clear=False, show=False, ax=ax)

    print("\tFORWARD[15] = ", forwardvals[15])
    print("\tBACKGROUND[15] = ", templvals[15])

    fullvals = templvals + forwardvals
    fullerrs = np.sqrt(np.square(templerrs) + np.square(forwarderrs))

    print("FULL[15] = ", fullvals[15])
    plotvalues(fullvals, fullerrs, h2reco.axes['dR'], "Total", dlog, savefig=None, clear=False, show=False, ax=ax)

    reco2vals, reco2cov = EECutil.getProjectedValsCov(h2reco, h2covreco, order=order,
                                                      flow=True, normalize=False)
    recovals = reco2vals[1:-1]
    recoerrs = np.sqrt(np.diag(reco2cov))[1:-1]
    print("RECO[15] = ", recovals[15])

    plotvalues(recovals, recoerrs, h2reco.axes['dR'], "Actual", dlog, savefig=savefig, clear=clear, show=show, ax=ax)

def plotBackground(htrans, hreco, hcovreco, hgen, hcovgen, order, 
                   label=None, dlog=True, savefig=None, clear=False, show=True,
                   ax=None):

    template, templatecov = EECutil.getBackgroundTemplate(htrans, 
                                                          hgen, hcovgen, 
                                                          hreco, hcovreco,
                                                          order = order,
                                                          normByGen=True)

    vals = template[1:-1]
    errs = np.sqrt(np.diag(templatecov))[1:-1]

    norm = EECutil.getForOrder1d(hgen, order).sum(flow=True)

    vals = vals*norm
    errs = errs*norm
    
    addYlabel(order, dlog, ax)
    plotvalues(vals, errs, hreco.axes['dR'], label, dlog, savefig, clear, show, ax)

def plotTransfered(htrans, hreco, order, label=None, dlog=True, savefig=None, clear=False, show=True, ax=None):
    orderidx = htrans.axes['order'].index(order)
    ht1d = htrans[{'order' : orderidx}].project('dRa')
    hr1d = hreco[{'order' : orderidx}].project('dR')

    norm = hr1d.sum(flow=True)
    vals = ht1d.values()/norm
    errs = 0

    addYlabel(order, dlog, ax)
    plotvalues(vals, errs, ht1d.axes['dRa'], label, dlog, savefig, clear, show, ax)

def plotProjectedEEC(h, hcov, order, label = None, dlog=True,
                     savefig=None, clear=False, show=True, ax=None, xlab=True):

    vals, cov = EECutil.getProjectedValsCov(h, hcov, order, normalize=False)
    errs = np.sqrt(np.diag(cov))
    
    addYlabel(order, dlog, ax)
    plotvalues(vals, errs, h.axes['dR'], label, dlog, savefig, clear, show, ax, xlab=xlab)

def plotProjectedDif(h1, h1cov, h2, h2cov, order, label=None, dlog=True,
             savefig=None, clear=False, show=True, ax=None):
    orderidx = h1.axes['order'].index(order)
    h11d = h1[{'order' : orderidx}].project('dR')
    h21d = h2[{'order' : orderidx}].project('dR')
    h1cov2d = h1cov[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')
    h2cov2d = h2cov[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')

    norm1 = h11d.sum(flow=True)
    norm2 = h21d.sum(flow=True)

    vals1 = h11d.values()/norm1
    errs1 = np.sqrt(np.diag(h1cov2d.values())/norm1)

    vals2 = h21d.values()/norm2
    errs2 = np.sqrt(np.diag(h2cov2d.values())/norm2)

    diff = vals1 - vals2
    differr = np.sqrt(np.square(errs1) + np.square(errs2))

    vals = diff/differr
    errs = 0

    if ax is None:
        ax = plt.gca()
    ax.set_ylabel("$\\frac{1-2}{\sigma}$", fontsize=20)
    
    low = np.min(h11d.axes['dR'].edges)
    high = np.max(h11d.axes['dR'].edges)
    x = np.linspace(low, high, 100)
    ax.fill_between(x, -1, 1, color='gray', alpha=0.5)
    ax.axhline(0, color='black', alpha=0.5, linestyle='--')
    print("filled between??")
    print("low, high", low, high)

    plotvalues(vals, errs, h11d.axes['dR'], label, dlog, savefig, clear, show, ax, False)

def plotProjectedWithRatio(h1, h1cov, h2, h2cov, order, dlog=True,
                           savefig=None, clear=False, show=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, 
                                         sharex=True)

    plotProjectedEEC(h1, h1cov, order, "1", dlog, None, False, False, ax1, xlab=False)
    plotProjectedEEC(h2, h2cov, order, "2", dlog, None, False, False, ax1, xlab=False)
    plotProjectedDif(h1, h1cov, h2, h2cov, order, None, dlog, savefig, clear, show, ax2)

def plotProjectedWithTransfer(htrans, hreco, hcovreco, hgen, hcovgen, order, dlog=True,
                              savefig=None, clear=False, show=True):
    plotProjectedEEC(hreco, hcovreco, order, "reco", dlog, show=False)
    plotBackground(htrans, hreco, hcovreco, hgen, hcovgen, order, 
                   "background", show=False)
    plotForward(htrans, hreco, hgen, hcovgen, order, 
                "transfered", dlog, savefig, clear, show)
