import pickle
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from probfit import Chi2Regression
from probfit.functor import Extended
from probfit.pdf import cruijff, gaussian
from iminuit import Minuit
import scipy.stats
import os.path
import probfit

def cutscan(H, cuts, prefix, etabin, baseDR, folder=None, postfix=''):
    #plt.clf()
    cmap = plt.get_cmap('viridis')
    for i, cut in enumerate(cuts):
        hist(H['%sCut%d%s'%(prefix,cut,postfix)][prefix]['dpt'], 'dpt', fit=False, label='%g'%(baseDR*cut), show=False, color=cmap(1-i/len(cuts)), etabin=etabin)
    if folder is not None:
        plt.savefig("%s/%s%s_eta%d_cutscan_dpt.png"%(folder,prefix,postfix,etabin), format='png', bbox_inches='tight')
    plt.show()

def histranges(h, axis, quantiles=[0.9, 0.92, 0.95, 0.99], ptmin=None):
    if ptmin is not None:
        ptidx = h.axes['pt'].index(ptmin)
        h = h[{'pt' : slice(ptidx,None,sum)}]

    h = h.project(axis)
    xs = np.concatenate(([-1000], h.axes[axis].centers, [1000]))
    ws = h.values(flow=True)
    
    ranges = []

    for quantile in quantiles:
        low = (1-quantile)/2
        high = 1 - low
        ranges.append(weightedquantile(xs, ws, high) 
                      - weightedquantile(xs, ws, low))
    
    ranges = np.asarray(ranges)

    if axis != 'dR':
        ranges = ranges/2

    return ranges

def argweightedquantile(xs, ws, quantile):
    Sn = ws.cumsum()
    return np.searchsorted(Sn, quantile * Sn[-1])

def weightedquantile(xs, ws, quantile):
    #thanks to https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median
    #we assume xs, ws is sorted in xs
    return xs[argweightedquantile(xs, ws, quantile)]

def trkresolution(pt, A, B):
    return pt*np.sqrt(np.square(A*pt) + np.square(B))

def caloresolution(pt, A, B):
    return np.sqrt(np.square(A/np.sqrt(pt)) + np.square(B))

def trkangularresolution(pt, A, B):
    return np.sqrt(np.square(A/np.power(pt, 1)) + np.square(B))

def caloangularresolution(pt, A):
    return np.ones_like(pt) * A

def std(H, axis='dphi'):
    x = H.axes[axis].centers
    w = H.project(axis).values(flow=False)
    return weightedquantile(x, w, scipy.stats.norm.cdf(1)) - weightedquantile(x, w, scipy.stats.norm.cdf(-1))

def mean(H, axis='dphi'):
    x = H.axes[axis].centers
    w = H.project(axis).values(flow=False)
    return weightedquantile(x, w, 0.5)

def dscb(x, norm, mu, sigma, alphaL, alphaR, nL, nR):
    return norm * probfit.vector_apply(probfit.pdf.doublecrystalball, x, alphaL, alphaR, nL, nR, mu, sigma)

def cruijff(x, norm, mu, sigmaL, sigmaR, alphaL, alphaR):
    return norm * probfit.vector_apply(probfit.pdf.cruijff, x, mu, sigmaL, sigmaR, alphaL, alphaR)

def fit_dscb(x, w):
    low = argweightedquantile(x, w, 0.05)
    high = argweightedquantile(x, w, 0.95)

    x = x[low:high]
    w = w[low:high]

    err = np.sqrt(w)

    norm0 = np.max(w)
    mu0 = np.average(x, weights=w)
    sigma0 = np.sqrt(np.sum(x*x*w)/np.sum(w))
    p0 = [norm0, mu0, sigma0, 1.0, 1.0, 2.0, 2.0]
    print("mu0 = %f" % mu0)
    print("sigma0 = %f" % sigma0)
    bounds = [(0, -np.inf, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)]
    popt, perr, infodict, mesg, ier = curve_fit(dscb, x, w, p0=p0, full_output=True, bounds=bounds, sigma=err)

    mu = popt[1]
    sigma = popt[2]

    plt.plot(x, dscb(x, *popt), label='fit')
    plt.text(0.01, 0.99, "$\mu=%0.2g$\n$\sigma=%0.2g$" % (mu, sigma), transform=plt.gca().transAxes, ha='left', va='top', fontsize=14)
    print(mu, sigma)

    return mu, sigma

def fit_cruijff(x, w):
    low = argweightedquantile(x, w, 0.005)
    high = argweightedquantile(x, w, 0.995)

    x = x[low:high]
    w = w[low:high]

    err = np.sqrt(w)

    norm0 = np.max(w)
    mu0 = np.average(x, weights=w)
    sigma0 = np.sqrt(np.sum(x*x*w)/np.sum(w))
    p0 = [norm0, mu0, sigma0, sigma0, 1.0, 1.0]
    print("mu0 = %f" % mu0)
    print("sigma0 = %f" % sigma0)
    bounds = [(0, -np.inf, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)]
    popt, perr, infodict, mesg, ier = curve_fit(cruijff, x, w, p0=p0, full_output=True, bounds=bounds, sigma=err)

    mu = popt[1]
    sigma = 0.5*(popt[2]+popt[3])

    plt.plot(x, cruijff(x, *popt), label='fit')
    plt.text(0.01, 0.99, "$\mu=%0.2g$\n$\sigma=%0.2g$" % (mu, sigma), transform=plt.gca().transAxes, ha='left', va='top', fontsize=14)
    print(mu, sigma)

    return mu, sigma

def etabinned(H, axis='dphi'):
    edges = H.axes['eta'].edges

    mus = []
    sigmas = []

    for i in range(len(edges)-1):
        H_i = H[{'eta' : i}]
        #print("%f < |eta| < %f:" % (edges[i], edges[i+1]))
        #print("\tmean: %f" % mean(H_i))
        #print("\tstd: %f" % std(H_i))
        mus.append(mean(H_i, axis))
        sigmas.append(std(H_i, axis))

    return mus, sigmas, edges

def ptetabinned(H, axis='dphi'):
    edges = H.axes['pt'].edges

    mus = []
    sigmas = []

    for i in range(len(edges)-1):
        H_i = H[{'pt' : i}]
        etamus, etasigmas, etaedges = etabinned(H_i, axis)
        mus.append(etamus)
        sigmas.append(etasigmas)

    return np.asarray(mus), np.asarray(sigmas), np.asarray(edges), np.asarray(etaedges)

def plot(H, axis='dphi', kind='trk'):
    mus, sigmas, ptedges, etaedges = ptetabinned(H, axis)

    ptcenters = (ptedges[1:] + ptedges[:-1]) / 2

    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True)
    fig.set_size_inches(12, 6)
    ax0.set_xlabel('$p_T$ [GeV]')
    ax1.set_xlabel('$p_T$ [GeV]')
    ax0.set_ylabel("Mean " + H.axes[axis].label)
    ax1.set_ylabel("RMS " + H.axes[axis].label)
    ax1.set_xscale('log')
    ax0.set_xscale('log')

    if kind=='trk':
        if axis=='dpt':
            func = trkresolution
        else:
            func = trkangularresolution
    elif kind=='calo':
        if axis == 'dpt':
            func = caloresolution
        else:
            func = caloangularresolution
    elif kind=='none':
        func = None
    else:
        raise ValueError("kind must be 'trk' or 'calo' or 'none'")

    for i in range(len(etaedges) - 1):
        c = next(ax0._get_lines.prop_cycler)['color']

        ax0.plot(ptcenters, mus[:,i], 'o', label='%f < |eta| < %f' % (etaedges[i], etaedges[i+1]), color=c)
        ax1.plot(ptcenters, sigmas[:,i], 'o', label='%f < |eta| < %f' % (etaedges[i], etaedges[i+1]), color=c)
        if func:
            popt, perr = curve_fit(func, ptcenters, sigmas[:,i])
            print(popt)
            ax1.plot(ptcenters, func(ptcenters, *popt), label=None, color=c)


    plt.tight_layout()
    plt.legend()
    plt.show()

def hist(H, axis='dpt', etabin=None, ptbinmin = None, ptbinmax = None, label=None, fname=None, show=True, fit=True, color=None, pdgid=None, density=False):
    x = H.axes[axis].centers
    edges = H.axes[axis].edges

    if etabin is not None:
        H = H[{'eta' : etabin}]
    if pdgid is not None:
        H = H[{'pdgid' : H.axes['pdgid'].index(pdgid)}]

    if axis != 'pt' and 'pt' in H.axes.name:
        w = H[{'pt' : slice(ptbinmin, ptbinmax, sum)}].project(axis).values(flow=False)
    else:
        w = H.project(axis).values(flow=False)

    x = x[1:-1]
    w = w[1:-1]
    edges = edges[1:-1]

    if density:
        w = w/H.sum().value

    plt.hist(x, bins=edges, weights=w, histtype='step', label=label, color=color, density=False)
    if fit:
        mu, sigma = fit_cruijff(x, w)
        #mu, sigma = fit_dscb(x, w)
    plt.xlabel(H.axes[axis].label)
    plt.ylabel("Events [arb. units]")

    if label is not None:
        plt.legend()

    if fname is not None:
        plt.savefig(fname, format='png', bbox_inches='tight')
        if not show:
            plt.clf()
    if show:
        plt.show()

    if fit:
        return mu, sigma
    else:
        return -1, -1

def pthists(H, axis='dpt', etabin=None, ptmin=None, ptmax=None, basepath=None, prefix=None, show=True, nstep=10, fit=True, kind='trk'):
    etaedges = H.axes['eta'].edges
    ptedges = H.axes['pt'].edges

    if ptmin is None:
        ptmin = np.min(ptedges)
    if ptmax is None:
        ptmax = np.max(ptedges)

    ptmin = H.axes['pt'].index(ptmin)
    ptmax = H.axes['pt'].index(ptmax)
    ptrange = ptmax - ptmin
    
    if nstep is None:
        nstep = ptrange

    ptlows = []
    pthighs = []
    mus = []
    sigmas = []

    for i in range(nstep):
        lowpt = int(ptmin + np.round(i * ptrange / nstep))
        highpt = int(ptmin + np.round((i+1) * ptrange / nstep))

        if basepath is not None:
            fname = os.path.join(basepath, '%s_ptbin_%d_%d.png' % (prefix, lowpt, highpt))
        else:
            fname = None

        plt.title("%0.2f < pT < %0.2f" % (ptedges[lowpt], ptedges[highpt]))

        mu, sigma = hist(H, axis=axis, etabin=etabin, ptbinmin=lowpt, ptbinmax=highpt, 
                         show=show, fname=fname, fit=fit)
        mus.append(mu),
        sigmas.append(sigma)
        ptlows.append(ptedges[lowpt])
        pthighs.append(ptedges[highpt])


    if nstep > 2 and fit:
        if kind=='trk':
            if axis=='dpt':
                func = trkresolution
            else:
                func = trkangularresolution
        elif kind=='calo':
            if axis == 'dpt':
                func = caloresolution
            else:
                func = caloangularresolution
        elif kind=='none':
            func = None
        else:
            raise ValueError("kind must be 'trk' or 'calo' or 'none'")


        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        ptlows = np.asarray(ptlows)
        pthighs = np.asarray(pthighs)
        ptmids = 0.5 * (ptlows + pthighs)

        plt.plot(ptmids, mus, 'o')
        plt.title("MU")
        plt.xscale('log')
        if basepath is not None:
            fname = os.path.join(basepath, '%s_mu.png' % (prefix))
            plt.savefig(fname, format='png', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
        plt.plot(ptmids, sigmas, 'o')
        if func is not None:
            popt, perr = curve_fit(func, ptmids, sigmas)
            plt.plot(ptmids, func(ptmids, *popt), label=None, color='k')
        plt.xscale('log')
        plt.title("SIGMA")
        if basepath is not None:
            fname = os.path.join(basepath, '%s_sigma.png' % (prefix))
            plt.savefig(fname, format='png', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()

        return ptmids, mus, sigmas


def etahists(H, axis='dpt', ptmin=None, ptmax=None, fname=None, show=True):
    etaedges = H.axes['eta'].edges
    ptedges = H.axes['pt'].edges

    if ptmin is not None:
        ptmin = H.axes['pt'].index(ptmin)
    if ptmax is not None:
        ptmax = H.axes['pt'].index(ptmax)

    for i in range(len(etaedges)-1):
        hist(H, axis, i, ptmin, ptmax, label='$%0.1f < |\eta| < %0.1f$' % (etaedges[i], etaedges[i+1]), show=False)

    plt.legend()

    plt.title("%0.2f < pT < %0.2f" % (ptedges[ptmin], ptedges[ptmax]))

    if fname is not None:
        plt.savefig(fname, format='png', bbox_inches='tight')
        if not show:
            plt.clf()
    if show:
        plt.show()

