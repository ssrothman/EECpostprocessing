import matplotlib.pyplot as plt
import hist
import numpy as np
import json
import pickle
import mplhep as hep
from plotting.util import *
import os

from samples.latest import SAMPLE_LIST

should_logx = {
    'nTruePU' : False,
    'Zpt' : True,
    'Zmass' : False,
    'Zy' : False,
    'MUpt' : True,
    'MUeta' : False,
    'MUpt_lead' : True,
    'MUeta_lead' : False,
    'MUpt_sub' : True,
    'MUeta_sub' : False,
    'NJet' : False,
    'Jpt' : True,
    'Jeta' : False,
    'HT' : True,
    'rho' : False,
    'btag_tight' : False,
    'btag_medium' : False,
    'btag_loose' : False,
    'METpt' : True,
    'METsig' : True,
    'METpt_nomask' : True,
    'METsig_nomask' : True,
    'nBtag' : False,
    'nBtag_nomask' : False,
}

should_logy = {
    'nTruePU' : True,
    'Zpt' : True,
    'Zmass' : True,
    'Zy' : True,
    'MUpt' : True,
    'MUeta' : True,
    'MUpt_lead' : True,
    'MUeta_lead' : True,
    'MUpt_sub' : True,
    'MUeta_sub' : True,
    'NJet' : True,
    'Jpt' : True,
    'Jeta' : True,
    'HT' : True,
    'rho' : True,
    'btag_tight' : True,
    'btag_medium' : True,
    'btag_loose' : True,
    'METpt' : True,
    'METsig' : True,
    'METpt_nomask' : True,
    'METsig_nomask' : True,
    'nBtag' : True,
    'nBtag_nomask' : True,
}

def getDataPU():
    with open("corrections/PU/PUhist2018UL.pkl", 'rb') as f:
        H = pickle.load(f)

    H2 = hist.Hist(
        hist.axis.Integer(0, 100, name='nTruePU', label='Number of True PU'),
    )
    H2 += H.values()

    return {'HPU': H2}

def plotAllKin(data, lumi, 
               bkgs, bkgxsecs, bkglabels,
               signals, signalxsecs, signallabels,
                   btag=None,
                   taglevel='tight',
                   normToLumi=True,
                   density=False,
                   dataname='Data',
                   show=True,
                   folder=None,
                   done=True):
    for whichhist in signals[0].keys():
        if 'sumwt' in whichhist or 'numjet' in whichhist:
            continue

        if data is not None and dataname=='Data' and 'HT' in whichhist:
            continue

        print(whichhist)
        for whichaxis in signals[0][whichhist].axes.name:
            if 'btag' in whichaxis:
                continue

            if 'True' in whichaxis and dataname == 'Data':
                logx = False
                logy = False

                if data is None:
                    HPU = None
                else:
                    HPU = getDataPU()

                plotKin(HPU, lumi, 
                        bkgs, bkgxsecs, bkglabels, 
                        signals, signalxsecs, signallabels,
                        whichhist, whichaxis,
                        logx=logx, logy=logy,
                        normToLumi=normToLumi,
                        density=True,
                        show=show,
                        done=done,
                        folder=folder,
                        dataname=dataname)
            else:
                logx = should_logx[whichaxis]
                logy = should_logy[whichaxis]

                plotKin(data, lumi, 
                        bkgs, bkgxsecs, bkglabels, 
                        signals, signalxsecs, signallabels,
                        whichhist, whichaxis,
                        btag=btag,
                        taglevel=taglevel,
                        logx=logx, logy=logy,
                        normToLumi=normToLumi,
                        done=done,
                        show=show,
                        folder=folder,
                        density=density,
                    dataname=dataname)

def plotKin(data, lumi, 
            bkgs, bkgxsecs, bkglabels,
            signals, signalxsecs, signallabels,
            whichhist, whichaxis, 
            btag=None,
            taglevel='tight',
            logx=False, logy=False,
            normToLumi=True,
            density=False,
            dataname='Data',
           show=True,
           folder=None,
           done=True):

    print(whichhist, whichaxis)

    if data is not None:
        fig, (ax0, ax1) = setup_ratiopad()
        add_cms_info(ax0, mcOnly=False)
    else:
        fig, ax0 = setup_plain()
        add_cms_info(ax0, mcOnly=True)

    if data is not None:
        if btag is not None and 'HJet' in whichhist:
            dataproj = data[whichhist][{'btag_%s'%taglevel : btag}].project(whichaxis)
        else:
            dataproj = data[whichhist].project(whichaxis)

        datavals = dataproj.values()
        if dataproj.variances() is not None:
            dataerrs = np.sqrt(dataproj.variances())
        else:
            dataerrs = np.zeros_like(datavals)

        if type(dataproj.axes[0]) is hist.axis.Integer:
            datacenters = dataproj.axes[0].centers - 0.5
            datawidths = dataproj.axes[0].widths
        else:
            datacenters = dataproj.axes[0].centers
            datawidths = dataproj.axes[0].widths

    bkghists = [mc[whichhist] for mc in bkgs]
    signalhists = [mc[whichhist] for mc in signals]
    if btag is not None and 'HJet' in whichhist:
        bkghists = [mc[{'btag_%s'%taglevel : btag}] for mc in bkghists]
        signalhists = [mc[{'btag_%s'%taglevel : btag}] for mc in signalhists]

    bkghists = [mc.project(whichaxis) for mc in bkghists]
    signalhists = [mc.project(whichaxis) for mc in signalhists]
    
    if type(signalhists[0].axes[0]) is hist.axis.Integer:
        mccenters = signalhists[0].axes[0].centers - 0.5
        mcwidths = signalhists[0].axes[0].widths
        mcedges = signalhists[0].axes[0].edges - 0.5
    else:
        mccenters = signalhists[0].axes[0].centers
        mcwidths = signalhists[0].axes[0].widths
        mcedges = signalhists[0].axes[0].edges

    bkgvals = [bkghist.values() for bkghist in bkghists]
    bkgerr2s = [bkghist.variances() for bkghist in bkghists]
    bkgsumwts = [bkg['sumwt'] for bkg in bkgs]

    signalvals = [signalhist.values() for signalhist in signalhists]
    signalerr2s = [signalhist.variances() for signalhist in signalhists]
    signalsumwts = [signal['sumwt'] for signal in signals]

    if normToLumi:
        bkgfactors = [1000 * lumi * xsec / sumwt for xsec, sumwt in zip(bkgxsecs, bkgsumwts)]
        signalfactors = [1000 * lumi * xsec / sumwt for xsec, sumwt in zip(signalxsecs, signalsumwts)]

        bkgvals = [bkgval * bkgfactor for bkgval, bkgfactor in zip(bkgvals, bkgfactors)]
        bkgerr2s = [bkgerr2 * bkgfactor**2 for bkgerr2, bkgfactor in zip(bkgerr2s, bkgfactors)]

        signalvals = [signalval * signalfactor for signalval, signalfactor in zip(signalvals, signalfactors)]
        signalerr2s = [signalerr2 * signalfactor**2 for signalerr2, signalfactor in zip(signalerr2s, signalfactors)]

    stacked_bkgvals = [np.zeros_like(bkgvals[0])]
    stacked_bkgerr2s = [np.zeros_like(bkgvals[0])]

    for bkgval, bkgerr2 in zip(bkgvals, bkgerr2s):
        stacked_bkgvals.append(stacked_bkgvals[-1] + bkgval)
        stacked_bkgerr2s.append(stacked_bkgerr2s[-1] + bkgerr2)

    stacked_signalvals = [stacked_bkgvals[-1] + signalval for signalval in signalvals]
    stacked_signalerr2s = [stacked_bkgerr2s[-1] + signalerr2 for signalerr2 in signalerr2s]

    if density and len(signalvals) > 1:
        raise ValueError("Cannot use density with multiple signals")

    if type(density) is str and density == 'proportion':
        N = stacked_signalvals[-1]
    elif density:
        N = stacked_signalvals[-1].sum()

    if density:
        stacked_signalvals = [signalval / N for signalval in stacked_signalvals]
        stacked_signalerr2s = [signalerr2 / (N*N) for signalerr2 in stacked_signalerr2s]

        stacked_bkgvals = [bkgval / N for bkgval in stacked_bkgvals]
        stacked_bkgerr2s = [bkgerr2 / (N*N) for bkgerr2 in stacked_bkgerr2s]

    if type(density) is str and density == 'proportion':
        pass 
    else:
        stacked_signalvals = [signalval / mcwidths for signalval in stacked_signalvals]
        stacked_signalerr2s = [signalerr2 / mcwidths**2 for signalerr2 in stacked_signalerr2s]

        stacked_bkgvals = [bkgval / mcwidths for bkgval in stacked_bkgvals]
        stacked_bkgerr2s = [bkgerr2 / mcwidths**2 for bkgerr2 in stacked_bkgerr2s]

    if len(signalvals) > 1: #then we have multiple signals, and should lump together all the backgrounds
        ax0.stairs(stacked_bkgvals[-1], mcedges, baseline=0, fill=True, label='Backgrounds', color='grey')

        for i in range(len(signals)):
            ax0.stairs(stacked_signalvals[i], mcedges, fill=False, label=signallabels[i])

    else: #then we resolve the full backgrounds stack
        for i in range(len(bkgs)):
            ax0.stairs(stacked_bkgvals[i+1], mcedges, 
                       baseline=stacked_bkgvals[i], 
                       fill=True, label=bkglabels[i])

        ax0.stairs(stacked_signalvals[0], mcedges, fill=True, baseline=stacked_bkgvals[-1], label="Signal")

    if data is not None:
        if type(density) is bool:
            N = datavals.sum()

        if density:
            datavals = datavals / N
            dataerrs = dataerrs / N

        if type(density) is str and density == 'proportion':
            pass
        else:
            datavals = datavals/datawidths
            dataerrs = dataerrs/datawidths

        #dataerrs = np.max(dataerrs, 0)
        ax0.errorbar(datacenters, datavals,
                     xerr = datawidths/2, yerr=dataerrs,
                     fmt='o', label=dataname, c='k')

    if logx:
        ax0.set_xscale('log')
    if logy:
        ax0.set_yscale('log')

    if data is not None:
        ax1.set_xlabel(signals[0][whichhist].axes[whichaxis].label)
    else:
        ax0.set_xlabel(signals[0][whichhist].axes[whichaxis].label)

    ylabelstr = ''

    if "pt" in whichaxis or 'mass' in whichaxis:
        ylabelstr += " Events / GeV"
    elif 'eta' or "y" in whichaxis:
        ylabelstr += " Events / unit"
    elif 'NJet' in whichaxis:
        ylabelstr += " Events"

    if type(density) is str and density == 'proportion':
        ylabelstr += ' [Proportion to total]'
    elif density:
        ylabelstr += ' [Density]'

    ax0.set_ylabel(ylabelstr)

    if data is not None:
        ax1.axhline(1, ls='--', c='r')

        if len(signalvals) == 1:
            ratio = datavals / stacked_signalvals[0]
            stacked_signalerrs = np.sqrt(stacked_signalerr2s[0])

            ratioerrs = ratio * np.sqrt(np.square(dataerrs/datavals) + \
                    np.square(stacked_signalerrs/stacked_signalvals[0]))

            ax1.errorbar(datacenters, ratio, 
                         xerr=datawidths/2, yerr=ratioerrs, 
                         fmt='o', c='k')
        else:
            ax1.axhline(1, ls='--', c='k')
            for i in range(len(signals)):
                
                ratio = datavals/ stacked_signalvals[i]
                stacked_signalerrs = np.sqrt(stacked_signalerr2s[i])

                ratioerrs = ratio * np.sqrt(np.square(dataerrs/datavals) + \
                        np.square(stacked_signalerrs/stacked_signalvals[i]))

                ax1.errorbar(datacenters, ratio, 
                             xerr=datawidths/2, yerr=ratioerrs, 
                             fmt='o')

        ax1.set_ylabel("Ratio")

        #ax1.set_ylim(0.8, 1.2)

    #ax0.legend()
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    #plt.savefig("kin/%s.png" % whichaxis, format='png', dpi=300, bbox_inches='tight')
    
    if folder is not None:
        savefig(os.path.join(folder, whichaxis+'.png'))

    if show:
        plt.show()
    elif done:
        plt.close()
    elif data is None:
        return fig, ax0
    else:
        return fig, (ax0, ax1)
