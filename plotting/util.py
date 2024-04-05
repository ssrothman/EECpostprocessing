import json
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

with open("configs/ak8.json", 'r') as f:
    config = json.load(f)

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
            edges = config['binning']['bins']['pt']
            i = bins[name]
            ansstr += '$%d < p_T$ [GeV] $< %d$'%(edges[i], edges[i+1])
        elif name == 'btag':
            if bins[name] == 0:
                ansstr += 'Fail b tag'
            else:
                ansstr += "pass b tag"
        else:
            print("WARNING UNSUPPORTED BINNING FOR TEXT")
        
        ansstr += '\n'
    return ansstr[:-1]

def getAXcenters_errs(ax):
    edges = np.asarray(config['binning']['bins'][ax])
    centers = 0.5 * (edges[1:] + edges[:-1])
    left = centers - edges[:-1]
    right = edges[1:] - centers
    return centers, (left, right)

def getDRcenters_errs():
    return getAXcenters_errs('dRedges')

def getPTcenters_errs():
    return getAXcenters_errs('pt')

def savefig(fname):
    plt.savefig(fname, format='png',
                bbox_inches='tight',
                dpi=300)

