import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import mplhep as hep

plt.style.use(hep.style.CMS)

def plotKin(data, lumi, mcs, xsecs, labels, whichhist, whichaxis, 
            logx=False, logy=False):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    hep.cms.text("Work in progress", ax=ax0)
    hep.cms.lumitext("$%0.2f fb^{-1}$ (13 TeV)" % (lumi), ax=ax0)

    datasumwt = data['Events']['sumwt']
    mcsumwts = [mc['Events']['sumwt'] for mc in mcs]

    dataproj = data['Events'][whichhist].project(whichaxis)
    datavals = dataproj.values()
    dataerrs = np.sqrt(dataproj.variances())
    datacenters = dataproj.axes[0].centers
    datawidths = dataproj.axes[0].widths

    mcprojs = [mc['Events'][whichhist].project(whichaxis) for mc in mcs]
    mcvals = [mcproj.values() for mcproj in mcprojs]
    mcerrs = [np.sqrt(mcproj.variances()) for mcproj in mcprojs]
    mccenters = [mcproj.axes[0].centers for mcproj in mcprojs]
    mcwidths = [mcproj.axes[0].widths for mcproj in mcprojs]
    mcedges = [mcproj.axes[0].edges for mcproj in mcprojs]

    vals = np.zeros_like(mcvals[0])
    err2s = np.zeros_like(mcvals[0])
    print(whichhist, whichaxis)
    for i in range(len(mcs)):
        #print(lumi)
        #print(xsecs[i])
        #mcvals[i] *= lumi * xsecs[i] / mcsumwts[i] * 1000
        #mcerrs[i] *= lumi * xsecs[i] / mcsumwts[i] * 1000
        newvals  = (mcvals[i] * lumi * xsecs[i] / mcsumwts[i] * 1000) / mcwidths[i]
        newerrs = (mcerrs[i] * lumi * xsecs[i] / mcsumwts[i] * 1000) / mcwidths[i]
        print("\t",labels[i], np.sum(mcvals[i] * lumi * xsecs[i] / mcsumwts[i]))
        #print(vals.sum())
        #print((vals+newvals).sum())
        #ax0.fill_between(mccenters[i], vals/1e6, (vals+newvals)/1e6, 
        #                 label=labels[i])
        ax0.stairs((vals+newvals)/1e6, mcedges[i], baseline=vals/1e6, 
                   fill=True, label=labels[i])
        vals += newvals
        err2s += np.square(newerrs)
        #plt.errorbar(mccenters[i], mcvals[i]/mcwidths[i],
        #             xerr = 0, yerr=mcerrs[i],
        #             fmt='o--', label=labels[i])

    print("data:", np.sum(datavals))
    datavals = datavals/datawidths
    dataerrs = dataerrs/datawidths

    ax0.errorbar(datacenters, datavals/1e6,
                 xerr = datawidths/2, yerr=dataerrs/1e6,
                 fmt='o', label='Data', c='k')

    if logx:
        ax0.set_xscale('log')
        ax1.set_xscale('log')
    if logy:
        ax0.set_yscale('log')

    ax1.set_xlabel(data['Events'][whichhist].axes[whichaxis].label)
    if "pt" in whichaxis or 'mass' in whichaxis:
        ax0.set_ylabel("Million Events / GeV")
    elif 'eta' or "y" in whichaxis:
        ax0.set_ylabel("Million Events / unit")
    elif 'NJet' in whichaxis:
        ax0.set_ylabel("Million Events")

    ratio = datavals / vals
    errs = np.sqrt(err2s)
    ratioerrs = ratio * np.sqrt(np.square(dataerrs/datavals) + np.square(errs/vals))

    ax1.axhline(1, ls='--', c='r')
    ax1.errorbar(datacenters, ratio, 
                 xerr=datawidths/2, yerr=ratioerrs, 
                 fmt='o--', c='k')
    ax1.set_ylabel("Data/MC")

    ax0.legend()
    plt.tight_layout()
    plt.savefig("kin/%s.png" % whichaxis, format='png', dpi=300, bbox_inches='tight')
    plt.show()

with open("Mar29_2024_nom_highstats/2018/SingleMuon/Kin/hists.pkl", 'rb') as f:
    data = pickle.load(f)

with open("Mar29_2024_nom_highstats/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/Pythia_ak8/Kin/hists.pkl", 'rb') as f:
    pythia = pickle.load(f)

with open("Mar29_2024_nom_highstats/2018/DYJetsToLL/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/Herwig_ak8/Kin/hists.pkl", 'rb') as f:
    herwig = pickle.load(f)

with open("./Mar29_2024_nom_highstats/2018/WW/Kin/hists.pkl", 'rb') as f:
    WW = pickle.load(f)

with open("./Mar29_2024_nom_highstats/2018/WZ/Kin/hists.pkl", 'rb') as f:
    WZ = pickle.load(f)

with open("./Mar29_2024_nom_highstats/2018/ZZ/Kin/hists.pkl", 'rb') as f:
    ZZ = pickle.load(f)

with open("./Mar29_2024_nom_highstats/2018/TT/Kin/hists.pkl", 'rb') as f:
    TT = pickle.load(f)

with open('configs/ak8.json') as f:
    config = json.load(f)

DYxsec = config['xsecs']['DYJetsToLL']
ZZxsec = config['xsecs']['ZZ']
WWxsec = config['xsecs']['WW']
WZxsec = config['xsecs']['WZ']
TTxsec = config['xsecs']['TTTo2L2Nu']
lumi = config['totalLumi']

should_logx = {
    'Zpt' : True,
    'Zmass' : False,
    'Zy' : False,
    'MUpt' : True,
    'MUeta' : False,
    'NJet' : False,
    'Jpt' : True,
    'Jeta' : False,
}

should_logy = {
    'Zpt' : True,
    'Zmass' : True,
    'Zy' : True,
    'MUpt' : True,
    'MUeta' : True,
    'NJet' : True,
    'Jpt' : True,
    'Jeta' : True,
}

for whichhist in data['Events'].keys():
    if whichhist == 'sumwt': continue
    for whichaxis in data['Events'][whichhist].axes.name:
        plotKin(data, lumi, 
                list(reversed([pythia, WW, TT, WZ, ZZ])), 
                list(reversed([DYxsec, WWxsec, TTxsec, WZxsec, ZZxsec])), 
                list(reversed(["DY", "WW", "TT", "WZ", "ZZ"])),
                whichhist, whichaxis,
                logx=should_logx[whichaxis],
                logy=should_logy[whichaxis])
