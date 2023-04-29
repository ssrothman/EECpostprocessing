import numpy as np
import awkward as ak
import hist
import matplotlib.pyplot as plt

def getPtAxis(name="pt", label="$p_{T}$ [GeV]", minpt=0.1, maxpt=100):
    return hist.axis.Regular(20, minpt, maxpt, name=name, label=label, transform=hist.axis.transform.log)

def getEtaAxis(name="eta", label="|$\eta$|"):
    return hist.axis.Regular(20, 0, 3, name=name, label=label, underflow=False)

def getPIDaxis(name='pdgid', label='pdgid'):
    return hist.axis.IntCategory([11, 13, 22, 130, 211], name=name, label=label)

def getMatchAxis(name='nmatch', label='nmatch'):
    return hist.axis.Integer(0, 5, name=name, label=label, underflow=False)

def getMatchHist():
    return hist.Hist(
        getPtAxis(),
        getEtaAxis(),
        getPIDaxis(),
        getMatchAxis(),
        getPtAxis("jetpt", "$p_{T, Jet}$ [GeV]", minpt=30, maxpt=1000),
        getPtAxis("fracpt", "$p_{T}/p_{T, Jet}$ [GeV]", minpt=0.01, maxpt=1),
        storage=hist.storage.Weight())

def fillMatchHist(h, parts, jets, evtwt=1):
    wt, jetpt, _ = ak.broadcast_arrays(evtwt, jets.pt, parts.pt)
    h.fill(pt = ak.flatten(parts.pt, axis=None),
           eta = np.abs(ak.flatten(parts.eta, axis=None)),
           pdgid = ak.flatten(parts.pdgid, axis=None),
           nmatch = ak.flatten(parts.nmatch, axis=None),
           jetpt = ak.flatten(jetpt, axis=None),
           fracpt = ak.flatten(parts.pt/jetpt, axis=None),
           weight = ak.flatten(wt, axis=None))

def plotMatchRate(h, var, ylabel='Particle matching rate'):
    h2d = h.project('nmatch', var)
    vals = h2d.values()
    variances = h2d.variances()
    
    norms = np.sum(vals, axis=0)

    pmiss = vals[0, :]/norms
    dpmiss = np.sqrt(variances[0, :])/norms

    varaxis = h.axes[var]

    plt.ylabel(ylabel)
    if(type(varaxis) == hist.axis.Regular):
        x = varaxis.centers
        if varaxis.transform is not None:
            plt.xscale('log')
        
        plt.errorbar(x, 1-pmiss, yerr=dpmiss, fmt='o')
        plt.xlabel(h.axes[var].label)
    elif type(varaxis) == hist.axis.IntCategory:
        x = np.arange(varaxis.size)
        plt.errorbar(x, 1-pmiss, yerr=dpmiss, fmt='o')
        plt.xticks(x, varaxis.value(x))
        plt.xlabel(h.axes[var].label)
    else:
        raise ValueError("Unknown axis type")
    plt.show()

