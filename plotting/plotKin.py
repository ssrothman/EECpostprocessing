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
    'NJet' : False,
    'Jpt' : True,
    'Jeta' : False,
    'HT' : True,
    'rho' : False,
}

should_logy = {
    'nTruePU' : False,
    'Zpt' : True,
    'Zmass' : True,
    'Zy' : True,
    'MUpt' : True,
    'MUeta' : True,
    'NJet' : True,
    'Jpt' : True,
    'Jeta' : True,
    'HT' : True,
    'rho' : True,
}

def getDataPU():
    with open("corrections/PU/PUhist2018UL.pkl", 'rb') as f:
        H = pickle.load(f)

    H2 = hist.Hist(
        hist.axis.Integer(0, 100, name='nTruePU', label='Number of True PU'),
    )
    H2 += H.values()

    return {'HPU': H2}

def plotAllKin(data, lumi, mcs, xsecs, labels,
                   normToLumi=True,
                   stack=True,
                   density=False,
                   dataname='Data',
                   show=True,
                   folder=None,
                   done=True):
    for whichhist in mcs[0].keys():
        if 'sumwt' in whichhist or 'numjet' in whichhist:
            continue

        if data is not None and dataname=='Data' and 'HT' in whichhist:
            continue

        print(whichhist)
        for whichaxis in mcs[0][whichhist].axes.name:
            if 'True' in whichaxis and dataname == 'Data':
                logx = False
                logy = False

                if data is None:
                    HPU = None
                else:
                    HPU = getDataPU()

                plotKin(HPU, lumi, mcs, xsecs, labels, whichhist, whichaxis,
                        logx=logx, logy=logy,
                        normToLumi=normToLumi,
                        stack=stack,
                        density=True,
                        show=show,
                        done=done,
                        folder=folder,
                        dataname=dataname)
            else:
                logx = should_logx[whichaxis]
                logy = should_logy[whichaxis]

                plotKin(data, lumi, mcs, xsecs, labels, whichhist, whichaxis,
                        logx=logx, logy=logy,
                        normToLumi=normToLumi,
                        stack=stack,
                        done=done,
                        show=show,
                        folder=folder,
                        density=density,
                    dataname=dataname)

def plotKin(data, lumi, mcs, xsecs, labels, whichhist, whichaxis, 
            logx=False, logy=False,
            normToLumi=True,
            stack = True,
            density=False,
            dataname='Data',
           show=True,
           folder=None,
           done=True):

    if data is not None:
        fig, (ax0, ax1) = setup_ratiopad()
        add_cms_info(ax0, mcOnly=False)
    else:
        fig, ax0 = setup_plain()
        add_cms_info(ax0, mcOnly=True)

    mcsumwts = [mc['sumwt'] for mc in mcs]

    if data is not None:
        dataproj = data[whichhist].project(whichaxis)
        #dataproj = data[whichhist]
        datavals = dataproj.values()
        if dataproj.variances() is not None:
            dataerrs = np.sqrt(dataproj.variances())
        else:
            dataerrs = np.zeros_like(datavals)

        datacenters = dataproj.axes[0].centers
        datawidths = dataproj.axes[0].widths

    mcprojs = [mc[whichhist].project(whichaxis) for mc in mcs]
    mcvals = [mcproj.values() for mcproj in mcprojs]
    mcerrs = [np.sqrt(mcproj.variances()) for mcproj in mcprojs]
    mccenters = [mcproj.axes[0].centers for mcproj in mcprojs]
    mcwidths = [mcproj.axes[0].widths for mcproj in mcprojs]
    mcedges = [mcproj.axes[0].edges for mcproj in mcprojs]

    if stack:
        vals = np.zeros_like(mcvals[0])
        err2s = np.zeros_like(mcvals[0])
    else:
        vals = []
        err2s = []

    print(whichhist, whichaxis)
    for i in range(len(mcs)):
        #print(lumi)
        #print(xsecs[i])
        #mcvals[i] *= lumi * xsecs[i] / mcsumwts[i] * 1000
        #mcerrs[i] *= lumi * xsecs[i] / mcsumwts[i] * 1000
        print("sumwt: ", mcsumwts[i])
        print("Before norm: ", labels[i], mcvals[i].sum())
        if normToLumi:
            newvals  = (mcvals[i] * lumi * xsecs[i] / mcsumwts[i] * 1000) 
            newerrs = (mcerrs[i] * lumi * xsecs[i] / mcsumwts[i] * 1000) 
        else:
            newvals = mcvals[i] 
            newerrs = mcerrs[i] 
        print("After norm: ", labels[i], newvals.sum())

        if density:
            N = newvals.sum()
            newvals = newvals / N
            newerrs = newerrs / N
        else:
            newvals /= 1e6
            newerrs /= 1e6
        print("After density: ", labels[i], newvals.sum())

        newvals = newvals / mcwidths[i]
        newerrs = newerrs / mcwidths[i]

        print("After width: ", labels[i], newvals.sum())

        print("\t",labels[i], np.sum(newvals*mcwidths[i]))
        #print(vals.sum())
        #print((vals+newvals).sum())
        #ax0.fill_between(mccenters[i], vals/1e6, (vals+newvals)/1e6, 
        #                 label=labels[i])
        if stack:
            ax0.stairs((vals+newvals), mcedges[i], baseline=vals, 
                       fill=True, label=labels[i])
            vals += newvals
            err2s += np.square(newerrs)
        else:
            vals += [newvals]
            err2s += [np.square(newerrs)]
            ax0.stairs(newvals, mcedges[i], baseline=0, fill=False, 
                       label=labels[i], lw=4)
        #plt.errorbar(mccenters[i], mcvals[i]/mcwidths[i],
        #             xerr = 0, yerr=mcerrs[i],
        #             fmt='o--', label=labels[i])

    if data is not None:
        print("data:", np.sum(datavals))
        if density:
            N = datavals.sum()
            datavals = datavals / N
            dataerrs = dataerrs / N
        else:
            datavals/=1e6
            dataerrs/=1e6

        datavals = datavals/datawidths
        dataerrs = dataerrs/datawidths

        ax0.errorbar(datacenters, datavals,
                     xerr = datawidths/2, yerr=dataerrs,
                     fmt='o', label=dataname, c='k')

    if logx:
        ax0.set_xscale('log')
    if logy:
        ax0.set_yscale('log')

    if data is not None:
        ax1.set_xlabel(mcs[0][whichhist].axes[whichaxis].label)
    else:
        ax0.set_xlabel(mcs[0][whichhist].axes[whichaxis].label)

    ylabelstr = ''
    if not density:
        ylabelstr += 'Million'

    if "pt" in whichaxis or 'mass' in whichaxis:
        ylabelstr += " Events / GeV"
    elif 'eta' or "y" in whichaxis:
        ylabelstr += " Events / unit"
    elif 'NJet' in whichaxis:
        ylabelstr += " Events"

    if density:
        ylabelstr += ' [Density]'

    ax0.set_ylabel(ylabelstr)

    if data is not None:
        if stack:
            ax1.axhline(1, ls='--', c='r')
            ratio = datavals / vals
            errs = np.sqrt(err2s)
            ratioerrs = ratio * np.sqrt(np.square(dataerrs/datavals) + np.square(errs/vals))

            ax1.errorbar(datacenters, ratio, 
                         xerr=datawidths/2, yerr=ratioerrs, 
                         fmt='o', c='k')
        else:
            ax1.axhline(1, ls='--', c='k')
            for val, err2 in zip(vals, err2s):
                ratio = datavals/ val
                errs = np.sqrt(err2)
                ratioerrs = ratio * np.sqrt(np.square(dataerrs/datavals) + np.square(errs/val))
                ax1.errorbar(datacenters, ratio, 
                             xerr=datawidths/2, yerr=ratioerrs, 
                             fmt='o', label=labels[i])
        ax1.set_ylabel("Ratio")

        #ax1.set_ylim(0.0, 2.0)

    ax0.legend()
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

#with open("configs/base.json", 'r') as f:
#    config = json.load(f)

#xsecs = config['xsecs-xsecDB']
#lumi = config['totalLumi']

#pythiaS = SAMPLE_LIST.lookup("DYJetsToLL_allHT")

#nom = pythiaS.get_hist('Kin')
#PUup = pythiaS.get_hist('Kin', ['PU_UP'])
#PUdn = pythiaS.get_hist('Kin', ['PU_DN'])

#dataS = SAMPLE_LIST.lookup("DATA_2018UL")
#data = dataS.get_hist('Kin')

#MCs = [nom, PUup, PUdn]
#labels = ['Nominal', 'PU up', 'PU down']
#MCxcs = [xsecs['DYJetsToLL']]*3

#fakePUdata = MCs[0]['HPU'].copy().reset()
#with open("corrections/PU/PUhist2018UL.pkl", 'rb') as f:
#    PUcorr = pickle.load(f)
#fakePUdata += PUcorr.values()
#fakePUdata = {'HPU': PUcorr,
#              'sumwt' : 0}

#WWS = SAMPLE_LIST.lookup("WW")
#WZS = SAMPLE_LIST.lookup("WZ")
#ZZS = SAMPLE_LIST.lookup("ZZ")
#TTS = SAMPLE_LIST.lookup("TTTo2L2Nu")

#bkgmcs = [WWS.get_hist('Kin'), 
#          ZZS.get_hist('Kin'), 
#          WZS.get_hist('Kin'),
#          TTS.get_hist('Kin'),
#          nom]
#bkgxsecs = [xsecs['WW'], 
#            xsecs['ZZ'], 
#            xsecs['WZ'], 
#            xsecs['TTTo2L2Nu'], 
#            xsecs['DYJetsToLL']]
#bkglabels = ['WW', 'ZZ', 'WZ', 'TT', 'DY']

#noRoccoR = pythiaS.get_hist('Kin', ['noRoccoR'])
#roccorMCs = [nom, noRoccoR]
#print(noRoccoR['sumwt'])
#roccorlabels = ['Nominal', 'No RoccoR']
#plotKin(data, lumi, roccorMCs, MCxcs, roccorlabels,
#        'HZ', 'Zmass',
#        stack=False,
#        density=False,
#        logx=True, logy=True)

#noSFs = pythiaS.get_hist('Kin', ['noIDsfs', 'noIsosfs', 'noTriggersfs'])
#noSFMCs = [nom, noSFs]
#noSFlabels = ['Nominal', 'No SFs']
##plotKin(data, lumi, noSFMCs, MCxcs, noSFlabels,
#        'Hmu', 'MUeta',
##        stack=False,
#        density=True,
#        logx=False, logy=True)

#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HZ', 'Zpt',
#        stack=True,
#        density=False,
#        logx=True, logy=True)
#
#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HZ', 'Zmass',
#        stack=True,
#        density=False,
#        logx=False, logy=True)

#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HZ', 'Zy',
#        stack=True,
#        density=False,
#        logx=False, logy=True)

#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HZ', 'Zpt',
#        stack=True,
#        density=False,
#        logx=True, logy=True)

#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HJet', 'Jpt',
#        stack=True,
#        density=False,
#        logx=True, logy=True)

#plotKin(data, lumi, bkgmcs, bkgxsecs, bkglabels,
#        'HJet', 'Jeta',
#        stack=True,
#        density=False,
#        logx=False, logy=True)

#plotKin(fakePUdata, lumi, MCs, MCxcs, labels, 
#        'HPU', 'nTruePU',
#        stack=False,
#        density=True)

#plotKin(data, lumi, MCs, MCxcs, labels, 
#        'Hrho', 'rho',
#        stack=False,
