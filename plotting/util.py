import json
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import json
import os
from RecursiveNamespace import RecursiveNamespace

plt.style.use(hep.style.CMS)

APPROVAL_TEXT = "Work in progress"

with open("configs/base.json", "r") as f:
    config = RecursiveNamespace(**json.load(f))

def apply_bins(H, bins):
    for ax in H.axes:
        if ax.name in bins.keys():
            H = H[{ax.name: bins[ax.name]}]
    return H

def setup_plain():
    return plt.subplots(1,1, figsize=(10,10))

def setup_ratiopad():
    return plt.subplots(2,1, figsize=(10,10), sharex=True,
                        gridspec_kw={'height_ratios': [3,1]})

def add_cms_info(ax, mcOnly):
    if mcOnly:
        hep.cms.text("Simulation %s"%APPROVAL_TEXT, ax=ax)
    else:
        hep.cms.text("%s"%APPROVAL_TEXT, ax=ax)

        lumi = config.totalLumi
        hep.cms.lumitext("%0.1ffb$^{-1}$ (13 TeV)"%lumi, ax=ax)

def binnedtext(bins):
    ansstr = ''
    for name in bins.keys():
        if name == 'order':
            if bins[name] == 0:
                ansstr += '2nd order'
            elif bins[name] == 1:
                ansstr += '3rd order'
            else:
                ansstr += '%dth order'%(bins[name]+2)
        elif name == 'pt':
            edges = config.binning.bins.pt
            i = bins[name]
            ansstr += '$%d < p_T$ [GeV] $< %d$'%(edges[i], edges[i+1])
        elif name == 'btag':
            if bins[name] == 0:
                ansstr += 'Fail b tag'
            else:
                ansstr += "pass b tag"
        elif name == 'eta':
            edges = config.binning.bins.eta
            i = bins[name]
            ansstr += '$%0.1f < \eta < %0.1f$'%(edges[i], edges[i+1])
        elif name == 'partCharge' : 
            if bins[name] == 0:
                ansstr += 'Neutrals'
            else:
                ansstr += 'Tracks'
        elif name == 'partSpecies' : 
            if bins[name] == 0:
                ansstr += 'Tracks'
            elif bins[name] == 1:
                ansstr += 'Photons'
            else:
                ansstr += 'Neutral Hadrons'
        else:
            print("WARNING UNSUPPORTED BINNING FOR TEXT")
        
        ansstr += '\n'
    return ansstr[:-1]

def getAXcenters_errs(ax):
    edges = np.asarray(getattr(config.binning.bins, ax))
    centers = 0.5 * (edges[1:] + edges[:-1])
    left = centers - edges[:-1]
    right = edges[1:] - centers
    return centers, (left, right)

def getDRcenters_errs():
    return getAXcenters_errs('dRedges')

def getPTcenters_errs():
    return getAXcenters_errs('pt')

def savefig(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, format='png',
                bbox_inches='tight',
                dpi=300)
