import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import transfer

def plotvalues(vals, errs, axis, label=None, dlog=True, savefig=None, clear=False, show=True):
    centers = axis.centers
    edges = axis.edges

    if dlog:
        widths = np.log(edges[1:]) - np.log(edges[:-1])
    else:
        widths = edges[1:] - edges[:-1]

    vals/=widths
    errs/=widths

    plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(centers, vals, yerr=errs, fmt='o--', label=label)
    plt.xlabel("$\Delta R_{max}$")
    plt.ylim(bottom=1e-4)

    if label is not None:
        plt.legend()

    if savefig is not None:
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
    if show:
        plt.show()
    if clear:
        plt.clf()

def addYlabel(order, dlog=True):
    if dlog:
        plt.ylabel("$\\frac{d\\sigma^{[%d]}}{d\\log\\Delta R_{max}}$"%order)
    else:
        plt.ylabel("$\\frac{d\\sigma^{[%d]}}{d\\Delta R_{max}}$"%order)

def plotForward(htrans, hreco, hgen, hcovgen, order,
                label=None, dlog=True, savefig=None, clear=False, show=True):
    orderidx = htrans.axes['order'].index(order)
    ht2d = htrans[{'order' : orderidx}].project('dRa', 'dRb')
    hr1d = hreco[{'order' : orderidx}].project('dR')
    hg1d = hgen[{'order' : orderidx}].project('dR')
    hgc2d = hcovgen[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')

    transMat = transfer.getTransferMatrix(ht2d, hg1d)
    reco, covreco = transfer.forwardTransfer(transMat, hg1d, hgc2d)
    reco = reco[1:-1]
    covreco = covreco[1:-1]

    norm = hr1d.sum(flow=True)
    vals = reco/norm
    errs = np.sqrt(np.diag(covreco))/norm

    addYlabel(order, dlog)
    plotvalues(vals, errs, hr1d.axes['dR'], label, dlog, savefig, clear, show)

def plotBackground(htrans, hreco, hcovreco, hgen, hcovgen, order, 
                   label=None, dlog=True, savefig=None, clear=False, show=True):
    orderidx = htrans.axes['order'].index(order)
    ht2d = htrans[{'order' : orderidx}].project('dRa', 'dRb')
    hr1d = hreco[{'order' : orderidx}].project('dR')
    hg1d = hgen[{'order' : orderidx}].project('dR')
    hgc2d = hcovgen[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')
    hrc2d = hcovreco[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')

    transMat = transfer.getTransferMatrix(ht2d, hg1d)
    reco, covreco = transfer.forwardTransfer(transMat, hg1d, hgc2d)
    reco = reco[1:-1]
    covreco = covreco[1:-1]

    norm = hr1d.sum(flow=True)
    valsFor = reco/norm
    errsFor = np.sqrt(np.diag(covreco))/norm

    valsReco = hr1d.values()/norm
    errsReco = np.sqrt(np.diag(hrc2d.values()))/norm

    vals = valsReco - valsFor
    errs = np.sqrt(errsReco**2 + errsFor**2)

    addYlabel(order, dlog)
    plotvalues(vals, errs, hr1d.axes['dR'], label, dlog, savefig, clear, show)

def plotTransfered(htrans, hreco, order, label=None, dlog=True, savefig=None, clear=False, show=True):
    orderidx = htrans.axes['order'].index(order)
    ht1d = htrans[{'order' : orderidx}].project('dRa')
    hr1d = hreco[{'order' : orderidx}].project('dR')

    norm = hr1d.sum(flow=True)
    vals = ht1d.values()/norm
    errs = 0

    addYlabel(order, dlog)
    plotvalues(vals, errs, ht1d.axes['dRa'], label, dlog, savefig, clear, show)

def plotProjectedEEC(h, hcov, order, label = None, dlog=True,
                     savefig=None, clear=False, show=True):

    orderidx = h.axes['order'].index(order)
    h1d = h[{'order' : orderidx}].project('dR')
    hcov2d = hcov[{'ordera' : orderidx, 'orderb' : orderidx}].project('dRa', 'dRb')

    norm = h1d.sum(flow=True)
    vals = h1d.values()/norm
    errs = np.diag(np.sqrt(hcov2d.values())/norm).copy()
    
    addYlabel(order, dlog)
    plotvalues(vals, errs, h1d.axes['dR'], label, dlog, savefig, clear, show)
