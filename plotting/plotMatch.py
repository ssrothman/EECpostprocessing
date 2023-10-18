import hist
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

def getpdgid(name):
    if 'EM0' in name:
        return 22
    elif 'HAD0' in name:
        return 130
    elif 'HADCH' in name:
        return 211
    elif 'ELE' in name:
        return 11
    elif 'MU' in name:
        return 13
    else:
        raise ValueError("Unknown prefix")

def cutscanefficiency(H, cuts, prefix, etabin, baseDR, ptval, folder=None, histname='matchReco'):
    pdgid = getpdgid(prefix)
    
    pmisses = []
    dpmisses = []
    for cut in cuts:
        pmiss, dpmiss = efficiencyAt(H['%sCut%dLimited'%(prefix, cut)][histname], ptval, pdgid=pdgid, etabin=etabin)
        pmisses.append(pmiss)
        dpmisses.append(dpmiss)

    plt.clf()
    plt.errorbar(baseDR*np.array(cuts), pmisses, yerr=dpmisses, fmt='o--')
    plt.xlabel("DR cut")
    plt.ylabel("Matching efficiency @ %g GeV"%ptval)

    if folder is not None:
        plt.savefig("%s/%s_eta%d_cutscan_%s_efficiency%d.png"%(folder,prefix,etabin,histname,ptval), format='png', bbox_inches='tight')
    plt.show()

def cutscan(H, cuts, prefix, etabin, baseDR, folder=None, postfix='', histname='matchReco'):
    pdgid = getpdgid(prefix)

    #plt.clf()
    cmap = plt.get_cmap('viridis')
    for i, cut in enumerate(cuts):
        plotMatchRate(H['%sCut%d%s'%(prefix,cut,postfix)][histname], 'pt', show=False, pdgid=pdgid, etabin=etabin, label='%g'%(baseDR*cut), color=cmap(1-i/len(cuts)))

    if folder is not None:
        plt.savefig("%s/%s%s_eta%d_cutscan_%s.png"%(folder,prefix,postfix,etabin,histname), format='png', bbox_inches='tight')
    plt.show()

def efficiencyAt(H, ptval, nmatch=-1, pdgid=None, etabin=None):
    if pdgid is not None:
        H = H[{'pdgid' : H.axes['pdgid'].index(pdgid)}]
    if etabin is not None:
        H = H[{'eta' : etabin}]

    h2d = H.project('nmatch', 'pt')
    vals = h2d.values(flow=True)
    variances = np.zeros_like(vals)#h2d.variances(flow=True)

    if H.axes['pt'].traits.underflow:
        vals = vals[:,1:]
        variances = variances[:,1:]
    if H.axes['pt'].traits.overflow:
        vals = vals[:,:-1]
        variances = variances[:,:-1]


    vals = vals[1:,:]
    variances = variances[1:,:]

    norms = np.sum(vals, axis=0)
    if(nmatch ==- 1):
        pmiss = 1-vals[0, :]/norms
        dpmiss = np.sqrt(variances[0, :])/norms
    else:
        pmiss = vals[nmatch, :]/norms
        dpmiss = np.sqrt(variances[nmatch, :])/norms

    ptidx = H.axes['pt'].index(ptval)
    return pmiss[ptidx], dpmiss[ptidx]



def plotMatchRate(h, var, nmatch = -1, ylabel='Matching Efficiency', 
                  savefig=None, clear=False, show=True, 
                  match='nmatch', label=None, pdgid=None, etabin=None,
                  color=None):
    if pdgid is not None:
        h = h[{'pdgid' : h.axes['pdgid'].index(pdgid)}]
    if etabin is not None:
        h = h[{'eta' : etabin}]
    h2d = h.project('nmatch', var)
    vals = h2d.values(flow=True)
    variances = np.zeros_like(vals)#h2d.variances(flow=True)

    if h.axes[var].traits.underflow:
        vals = vals[:,1:]
        variances = variances[:,1:]
    if h.axes[var].traits.overflow:
        vals = vals[:,:-1]
        variances = variances[:,:-1]

    vals = vals[1:,:]
    variances = variances[1:,:]

    norms = np.sum(vals, axis=0)

    if(nmatch ==- 1):
        pmiss = 1-vals[0, :]/norms
        dpmiss = np.sqrt(variances[0, :])/norms
    else:
        pmiss = vals[nmatch, :]/norms
        dpmiss = np.sqrt(variances[nmatch, :])/norms


    varaxis = h.axes[var]

    plt.ylabel(ylabel)
    plt.ylim(0, 1.1)
    plt.axhline(y=1, color='black', linestyle='--')
    if(type(varaxis) == hist.axis.Regular):
        x = varaxis.centers
        if varaxis.transform is not None:
            plt.xscale('log')
        
        plt.errorbar(x, pmiss, yerr=dpmiss, fmt='o--', label=label, color=color)
        plt.xlabel(h.axes[var].label)
    elif type(varaxis) == hist.axis.IntCategory:
        x = np.arange(varaxis.size)
        plt.errorbar(x, pmiss, yerr=dpmiss, fmt='o', label=label, color=color)
        plt.xticks(x, varaxis.value(x))
        plt.xlabel(h.axes[var].label)
    else:
        raise ValueError("Unknown axis type")

    if label is not None:
        plt.legend()

    if savefig is not None:
        plt.savefig(savefig+'.png', bbox_inches='tight', format='png')
    if show:
        plt.show()
    if clear:
        plt.clf()

def pdgIdPlot(h, var, nmatch=-1, etabin=None, ylabel='Particle matching rate', 
              savefig=None, clear=False, show=True, 
              match='nmatch'):
    plotMatchRate(h, var, nmatch, ylabel, savefig, False, False, False, 'Neutral EM', 22, etabin=etabin)
    plotMatchRate(h, var, nmatch, ylabel, savefig, False, False, False, 'Neutral HAD', 130, etabin=etabin)
    plotMatchRate(h, var, nmatch, ylabel, savefig, False, False, False, 'Charged HAD', 211, etabin=etabin)
    plotMatchRate(h, var, nmatch, ylabel, savefig, False, False, False, 'Electrons', 11, etabin=etabin)
    plotMatchRate(h, var, nmatch, ylabel, savefig, clear, show, match, 'Muons', 13, etabin=etabin)
