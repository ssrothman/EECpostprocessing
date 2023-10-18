import post.resolution
import itertools
import plotting.plotMatch

import matplotlib.pyplot as plt

AOD = {}
miniAOD = {}

def set_dicts(x):
    global AOD 
    AOD = x.AOD
    global miniAOD 
    miniAOD = x.miniAOD

#post.resolution.cutscan(AOD, [1,2,3,5,7,10,15,20,40], 'EM0', 0, 0.01, postfix='Drop')
#post.resolution.cutscan(miniAOD, [1,2,3,5,7,10,15,20,40], 'EM0', 0, 0.01, postfix='Drop')
piddict = {'EM0' : 22, 'HAD0' : 130, 'HADCH' : 211, 'ELE' : 11, 'MU' : 13}

def cutscan(pid, cuts, suffix, baseDR, doAOD, folder=None):
    if doAOD:
        D = AOD
    else:
        D = miniAOD

    pdgid = piddict[pid]


    for etabin, name in zip([0, 1, 2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        plt.title("%s resolution"%name)
        post.resolution.cutscan(D, cuts, pid, etabin, baseDR, postfix=suffix, folder=folder)

        plt.title("%s reco efficiency"%name)
        plotting.plotMatch.cutscan(D, cuts, pid, etabin, baseDR, postfix=suffix, histname='matchReco', folder=folder)
        
        plt.title("%s gen efficiency"%name)
        plotting.plotMatch.cutscan(D, cuts, pid, etabin, baseDR, postfix=suffix, histname='matchGen', folder=folder)

def compare_suffix(pid, cut, suffixes, labels, doAOD, ptbin=None, folder=None):
    if doAOD:
        D = AOD
    else:
        D = miniAOD

    pdgid = piddict[pid]

    ptbinmin = ptbin
    if ptbinmin is None:
        ptbinmax = None
    else:
        ptbinmax = ptbin+1

    if folder is not None:
        fnamesuffix = ''
        for suffix in suffixes:
            fnamesuffix += '%s_vs_'%suffix
        fnamesuffix = fnamesuffix[:-4]
        fnamesuffix += '_cut%d'%cut

    for etabin, name in zip([0, 1, 2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        plt.title("%s resolution"%name)
        for suffix, label in zip(suffixes, labels):
            post.resolution.hist(D['%sCut%d%s'%(pid,cut,suffix)][pid]['dpt'], 'dpt', etabin=etabin, label=label, show=False, fit=False, density=True, ptbinmin=ptbinmin, ptbinmax=ptbinmax)
        if folder is not None:
            plt.savefig('%s/%s_eta%d_%s_dpt.png'%(folder,pid,etabin,fnamesuffix), format='png', bbox_inches='tight')
        plt.show()

        plt.title("%s reco efficiency"%name)
        for suffix, label in zip(suffixes, labels):
            plotting.plotMatch.plotMatchRate(D['%sCut%d%s'%(pid,cut,suffix)]['matchReco'], 'pt', etabin=etabin, label=label, show=False, pdgid=pdgid)
        if folder is not None:
            plt.savefig('%s/%s_eta%d_%s_matchReco.png'%(folder,pid,etabin,fnamesuffix), format='png', bbox_inches='tight')
        plt.show()

        plt.title("%s gen efficiency"%name)
        for suffix, label in zip(suffixes, labels):
            plotting.plotMatch.plotMatchRate(D['%sCut%d%s'%(pid,cut,suffix)]['matchGen'], 'pt', etabin=etabin, label=label, show=False, pdgid=pdgid)
        if folder is not None:
            plt.savefig('%s/%s_eta%d_%s_matchGen.png'%(folder,pid,etabin,fnamesuffix), format='png', bbox_inches='tight')
        plt.show()

def compare_AOD_kinnematics(pid, cut, suffix, histname, folder=None):
    pdgid = piddict[pid]

    plt.title("Barrel")
    post.resolution.hist(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=0, label='AOD', show=False, fit=False, pdgid=pdgid, density=True)
    post.resolution.hist(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=0, label='miniAOD', fit=False, pdgid=pdgid, density=True, show=False)
    plt.xscale('log')
    if folder is not None:
        plt.savefig('%s/%s%s_eta0_compareAOD_%s_pt.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

    plt.title("Barrel/Endcap Transition")
    post.resolution.hist(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=1, label='AOD', show=False, fit=False, pdgid=pdgid, density=True)
    post.resolution.hist(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=1, label='miniAOD', fit=False, pdgid=pdgid, density=True, show=False)
    plt.xscale('log')
    if folder is not None:
        plt.savefig('%s/%s%s_eta1_compareAOD_%s_pt.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

    plt.title("Endcaps")
    post.resolution.hist(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=2, label='AOD', show=False, fit=False, pdgid=pdgid, density=True)
    post.resolution.hist(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', etabin=2, label='miniAOD', fit=False, pdgid=pdgid, density=True, show=False)
    plt.xscale('log')
    if folder is not None:
        plt.savefig('%s/%s%s_eta2_compareAOD_%s_pt.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

def compare_AOD_efficiency(pid, cut, suffix, histname, folder=None):
    pdgid = piddict[pid]

    plt.title("Barrel")
    plotting.plotMatch.plotMatchRate(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=0, label='AOD', show=False)
    plotting.plotMatch.plotMatchRate(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=0, label='miniAOD', show=False)
    if folder is not None:
        plt.savefig('%s/%s%s_eta0_compareAOD_%s.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

    plt.title("Barrel/Endcap Transition")
    plotting.plotMatch.plotMatchRate(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=1, label='AOD', show=False)
    plotting.plotMatch.plotMatchRate(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=1, label='miniAOD', show=False)
    if folder is not None:
        plt.savefig('%s/%s%s_eta1_compareAOD_%s.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

    plt.title("Endcap")
    plotting.plotMatch.plotMatchRate(AOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=2, label='AOD', show=False)
    plotting.plotMatch.plotMatchRate(miniAOD['%sCut%d%s'%(pid,cut,suffix)][histname], 'pt', pdgid=pdgid, etabin=2, label='miniAOD', show=False)
    if folder is not None:
        plt.savefig('%s/%s%s_eta2_compareAOD_%s.png'%(folder,pid,suffix,histname), format='png', bbox_inches='tight')
    plt.show()

def compare_filters(elecuts, phocuts, drops, recovers, labels, doAOD=True, folder=None, show=True, pids=['ELE', 'MU', 'HAD0', 'EM0', 'HADCH']):
    if type(elecuts) not in [list, tuple]:
        elecuts = [elecuts]
    if type(phocuts) not in [list, tuple]:
        phocuts = [phocuts]
    if type(drops) not in [list, tuple]:
        drops = [drops]
    if type(recovers) not in [list, tuple]:
        recovers = [recovers]

    maxlen = max(len(elecuts), len(phocuts), len(drops), len(recovers))
    if len(elecuts) < maxlen:
        if len(elecuts) == 1:
            elecuts = elecuts*maxlen
        else:
            raise ValueError("elecuts must be a list of the same length as phocuts, drops, and recovers, or else have length 1")
    if len(phocuts) < maxlen:
        if len(phocuts) == 1:
            phocuts = phocuts*maxlen
        else:
            raise ValueError("phocuts must be a list of the same length as elecuts, drops, and recovers, or else have length 1")
    if len(drops) < maxlen:
        if len(drops) == 1:
            drops = drops*maxlen
        else:
            raise ValueError("drops must be a list of the same length as elecuts, phocuts, and recovers, or else have length 1")
    if len(recovers) < maxlen:
        if len(recovers) == 1:
            recovers = recovers*maxlen
        else:
            raise ValueError("recovers must be a list of the same length as elecuts, phocuts, and drops, or else have length 1")
    if len(labels) != maxlen:
        raise ValueError("labels must be a list of the same length as elecuts, phocuts, drops, and recovers")

    if doAOD:
        D = AOD
    else:
        D = miniAOD

    for etabin, title in zip([0,1,2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        for pid in pids:
            for histname in ['matchReco', 'matchGen']:
                plt.title(pid + " " + title + " " + histname)
                for elecut, phocut, drop, recover, label in zip(elecuts, phocuts, drops, recovers, labels):
                    name_ele = '%sEle'%elecut
                    name_pho = '%sPho'%phocut
                    name_drop = 'Drop' if drop else 'NoDrop'
                    name_recover = 'Recover' if recover else 'Norecover'
                    fullname = '%s%s%s%s'%(name_ele, name_pho, name_drop, name_recover)
                    plotting.plotMatch.plotMatchRate(D[fullname][histname], 'pt', 
                            pdgid=piddict[pid], etabin=etabin, 
                            label=label, show=False)
                if folder is not None:
                    plt.savefig('%s/%s_eta%d_compareFilters_%s.png'%(folder,pid,etabin,histname), format='png', bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.clf()

            plt.title(pid + " " + title + " " + "dpt\n pt < 4 GeV")
            for elecut, phocut, drop, recover, label in zip(elecuts, phocuts, drops, recovers, labels):
                name_ele = '%sEle'%elecut
                name_pho = '%sPho'%phocut
                name_drop = 'Drop' if drop else 'NoDrop'
                name_recover = 'Recover' if recover else 'Norecover'
                fullname = '%s%s%s%s'%(name_ele, name_pho, name_drop, name_recover)
                post.resolution.hist(D[fullname][pid]['dpt'], 'dpt',
                        etabin=etabin, density=False, fit=False,
                        label=label, show=False,
                        ptbinmin=0, ptbinmax=1)
            if folder is not None:
                plt.savefig('%s/%s_eta%d_compareFilters_soft_dpt.png'%(folder,pid,etabin), format='png', bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.clf()

            plt.title(pid + " " + title + " " + "dpt\n pt > 8 GeV")
            for elecut, phocut, drop, recover, label in zip(elecuts, phocuts, drops, recovers, labels):
                name_ele = '%sEle'%elecut
                name_pho = '%sPho'%phocut
                name_drop = 'Drop' if drop else 'NoDrop'
                name_recover = 'Recover' if recover else 'Norecover'
                fullname = '%s%s%s%s'%(name_ele, name_pho, name_drop, name_recover)
                post.resolution.hist(D[fullname][pid]['dpt'], 'dpt',
                        etabin=etabin, density=False, fit=False,
                        label=label, show=False,
                        ptbinmin=2, ptbinmax=None)
            if folder is not None:
                plt.savefig('%s/%s_eta%d_compareFilters_hard_dpt.png'%(folder,pid,etabin), format='png', bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.clf()


            

def plot_filter(elecut, phocut, drop, recover, doAOD=True, folder=None):
    name_ele = '%sEle'%elecut
    name_pho = '%sPho'%phocut
    name_drop = 'Drop' if drop else 'NoDrop'
    name_recover = 'Recover' if recover else 'Norecover'
    fullname = '%s%s%s%s'%(name_ele, name_pho, name_drop, name_recover)

    if doAOD:
        D = AOD
    else:
        D = miniAOD

    for etabin, title in zip([0,1,2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        for histname in ['matchReco', 'matchGen']:
            plt.title(title + " " + histname)
            plotting.plotMatch.pdgIdPlot(D[fullname][histname], 'pt', etabin=etabin,
                 savefig='%s/%s_eta%d_%s'%(folder,fullname,etabin,histname))

        for pid in 'HAD0', 'EM0', 'HADCH', 'ELE', 'MU':
            plt.title(title + " %s"%pid)

            pdgid = piddict[pid]
            post.resolution.hist(D[fullname][pid]['dpt'], 'dpt', 
                 etabin=etabin, fit=False,
                 fname='%s/%s_eta%d_%s_dpt.png'%(folder, fullname, etabin, pid)) 

def jecplots(doAOD=True, folder=None):
    if doAOD:
        D = AOD
    else:
        D = miniAOD

    H = D['LooseNoMuEleTightPhoDropRecover']['jets']

    for etabin, title in zip([0,1,2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        for ptmin, ptmax in zip([0, 4], [2, 10]):
            ptminval = H['jecRatio'].axes['pt'].edges[ptmin]
            ptmaxval = H['jecRatio'].axes['pt'].edges[ptmax]

            plt.title(title + "\n %d < pT [GeV] < %d"%(ptminval, ptmaxval))
            for term, label in zip(['responseTerm', 'puTerm', 'unrecoTerm', 'jecPred'],
                                   ['Energy response', 'Pileup', 'Missed gen', 'Total']):
                post.resolution.hist(H[term], term, label=label,
                                     etabin=etabin, fit=False, show=False)
            if folder is not None:
                plt.savefig('%s/jec_eta%d_pt%dto%d.png'%(folder,etabin, ptmin, ptmax), format='png', bbox_inches='tight')
            plt.show()

def plotgivenjecs(doAOD=True, folder=None):
    if doAOD:
        D = AOD
    else:
        D = miniAOD

    H = D['LooseNoMuEleTightPhoDropRecover']['jets']

    for etabin, title in zip([0,1,2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        for ptmin, ptmax in zip([0, 4], [2, 10]):
            ptminval = H['jecRatio'].axes['pt'].edges[ptmin]
            ptmaxval = H['jecRatio'].axes['pt'].edges[ptmax]

            plt.title(title + "\n %d < pT [GeV] < %d"%(ptminval, ptmaxval))
            plt.axvline(1, color='k', linestyle='--')
            post.resolution.hist(H['jecRatio'], 'jecRatio', etabin=etabin, 
                                 fit=False, show=True,
                                 ptbinmin = ptmin, ptbinmax = ptmax,
                                 fname='%s/jer_eta%d_pt%dto%d.png'%(folder,etabin, ptmin, ptmax) if folder is not None else None)

            plt.title(title + "\n %d < pT [GeV] < %d"%(ptminval, ptmaxval))
            plt.axvline(1, color='k', linestyle='--')
            post.resolution.hist(H['jec'][{'pt' : slice(ptmin, ptmax, sum)}], 'jec', etabin=etabin, 
                                 fit=False, show=True,
                                 ptbinmin = ptmin, ptbinmax = ptmax,
                                 fname='%s/jec_cmssw_eta%d_pt%dto%d.png'%(folder,etabin, ptmin, ptmax) if folder is not None else None)

def compare_filters_oct12(hadchfilters, had0filters,
                          elefilters, em0filters,
                          drops, chargefilters, 
                          recovers, thresholds, 
                          labels, 
                          folder=None, show=True, 
                          pids=['ELE', 'MU', 'HAD0', 'EM0', 'HADCH']):
    if type(hadchfilters) not in [list, tuple]:
        hadchfilters = [hadchfilters]
    if type(had0filters) not in [list, tuple]:
        had0filters = [had0filters]
    if type(elefilters) not in [list, tuple]:
        elefilters = [elefilters]
    if type(em0filters) not in [list, tuple]:
        em0filters = [em0filters]
    if type(drops) not in [list, tuple]:
        drops = [drops]
    if type(chargefilters) not in [list, tuple]:
        chargefilters = [chargefilters]
    if type(recovers) not in [list, tuple]:
        recovers = [recovers]
    if type(thresholds) not in [list, tuple]:
        thresholds = [thresholds]

    maxlen = max(len(hadchfilters), len(drops), 
                 len(chargefilters), len(recovers), 
                 len(thresholds), len(had0filters), 
                 len(elefilters), len(em0filters))
    if len(hadchfilters) == 1:
        hadchfilters = hadchfilters * maxlen
    if len(drops) == 1:
        drops = drops * maxlen
    if len(chargefilters) == 1:
        chargefilters = chargefilters * maxlen
    if len(recovers) == 1:
        recovers = recovers * maxlen
    if len(thresholds) == 1:
        thresholds = thresholds * maxlen
    if len(had0filters) == 1:
        had0filters = had0filters * maxlen
    if len(elefilters) == 1:
        elefilters = elefilters * maxlen
    if len(em0filters) == 1:
        em0filters = em0filters * maxlen

    D = AOD

    etabins = [0,1,2]
    titles = ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']
    for etabin, title in zip(etabins, titles):
        for pid in pids:
            for histname in ['matchReco', 'matchGen']:
                plt.title(pid + " " + title + " " + histname)
                for i in range(maxlen):
                    label = labels[i]
                    hadchfilter = hadchfilters[i]
                    had0filter = had0filters[i]
                    elefilter = elefilters[i]
                    em0filter = em0filters[i]
                    drop = drops[i]
                    chargefilter = chargefilters[i]
                    recover = recovers[i]
                    threshold = thresholds[i]

                    fullname = '%s%s%s%s%s%s%s%s'%(hadchfilter, had0filter, 
                                                   elefilter, em0filter,
                                                   drop, chargefilter, 
                                                   recover, threshold)
                    print(fullname)
                    plotting.plotMatch.plotMatchRate(D[fullname][histname], 'pt', 
                            pdgid=piddict[pid], etabin=etabin, 
                            label=label, show=False)
                if folder is not None:
                    plt.savefig('%s/%s_eta%d_compareFilters_%s.png'%(folder,pid,etabin,histname), format='png', bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.clf()

            for ptbinmin, ptbinmax in zip([0, 2], [1, 25]):
                testname = '%s%s%s%s%s%s%s%s'%(hadchfilters[0], had0filters[0],
                                               elefilters[0], em0filters[0],
                                               drops[0], chargefilters[0],
                                               recovers[0], thresholds[0])
                ptminval = D[testname]['HADCH']['dpt'].axes['pt'].edges[ptbinmin]
                ptmaxval = D[testname]['HADCH']['dpt'].axes['pt'].edges[ptbinmax]
                print()
                print(ptbinmin, '->', ptminval)
                print(ptbinmax, '->', ptmaxval)
                print()
                plt.title(pid + " " + title + " " + "dpt\n %d < pt < %d GeV"%(ptminval, ptmaxval))
                for i in range(maxlen):
                    label = labels[i]
                    hadchfilter = hadchfilters[i]
                    had0filter = had0filters[i]
                    elefilter = elefilters[i]
                    em0filter = em0filters[i]
                    drop = drops[i]
                    chargefilter = chargefilters[i]
                    recover = recovers[i]
                    threshold = thresholds[i]

                    fullname='%s%s%s%s%s%s%s%s'%(hadchfilter, had0filter,
                                                 elefilter, em0filter,
                                                 drop, chargefilter,
                                                 recover, threshold)
                    print(fullname)

                    post.resolution.hist(D[fullname][pid]['dpt'], 'dpt',
                            etabin=etabin, density=False, fit=False,
                            label=label, show=False,
                            ptbinmin=ptbinmin, ptbinmax=ptbinmax)
                if folder is not None:
                    plt.savefig('%s/%s_eta%d_compareFilters_pt%dto%d_dpt.png'%(folder,pid,etabin, ptbinmin, ptbinmax), format='png', bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.clf()

def study_neutralflavorfilters(show=True):
    compare_filters_oct12('Tight',
                          'Tight',
                          'Tight', 
                          ['Tight', 'Loose'],
                          'No', 'Any', 'No', 'No',
                          labels = ['Tight ECAL', 'Loose ECAL'],
                          pids = ['HAD0', 'EM0'],
                          folder = 'plots_oct12/NeutralFlavorFilters/ECAL',
                          show=show)

    compare_filters_oct12('Tight',
                          ['Tight', 'Loose'],
                          'Tight',
                          'Tight', 
                          'No', 'Any', 'No', 'No',
                          labels = ['Tight HCAL', 'Loose HCAL'],
                          pids = ['HAD0', 'EM0'],
                          folder = 'plots_oct12/NeutralFlavorFilters/HCAL',
                          show=show)

def study_chargedfilters(show=True):
    compare_filters_oct12(['Tight', 'Loose', 'Loose'],
                          'Tight',
                          'Tight',
                          'Tight',
                          'No', 
                          ['Any', 'Any', 'Tight'],
                          'No',
                          'No',
                          labels = ['Tight flavor match', 'Allow electrons', 
                                    'Allow electrons + tight charge matching'],
                          pids=['HADCH', 'ELE'],
                          folder='plots_oct12/ChargedFilters/HADCH',
                          show=show)
    
    compare_filters_oct12('Tight',
                          'Tight',
                          ['Tight', 'Loose', 'Loose'],
                          'Tight',
                          'No', 
                          ['Any', 'Any', 'Tight'],
                          'No',
                          'No',
                          labels = ['Tight flavor match', 'Allow electrons', 
                                    'Allow electrons + tight charge matching'],
                          pids=['HADCH', 'ELE'],
                          folder='plots_oct12/ChargedFilters/ELE',
                          show=show)

def study_recover(show=True):
    compare_filters_oct12('Loose', 'Tight', 'Loose', 'Loose',
                          'No', 'Tight', ['No', 'Limited', 'Free'], 'No',
                          labels = ['None', 'Thresholded', 'Full'],
                          pids = ['HADCH', 'ELE', 'EM0', 'HAD0'],
                          folder='plots_oct12/Recover/TightHAD0',
                          show=show)

    compare_filters_oct12('Loose', 'Loose', 'Loose', 'Loose',
                          'No', 'Tight', ['No', 'Limited', 'Free'], 'No',
                          labels = ['None', 'Thresholded', 'Full'],
                          pids = ['HADCH', 'ELE', 'EM0', 'HAD0'],
                          folder='plots_oct12/Recover/LooseHAD0',
                          show=show)

def study_thresholds(show=True):
    compare_filters_oct12('Loose', 'Tight', 'Loose', 'Loose',
                          'No', 'Tight', 'Limited', ['Yes', 'No'],
                          labels = ['Thresholded', 'Not thresholded'],
                          folder='plots_oct12/Thresholds/TightHAD0',
                          show=show)

    compare_filters_oct12('Loose', 'Loose', 'Loose', 'Loose',
                          'No', 'Tight', 'Limited', ['Yes', 'No'],
                          labels = ['Thresholded', 'Not thresholded'],
                          folder='plots_oct12/Thresholds/LooseHAD0',
                          show=show)

def study_drops(show=True):
    compare_filters_oct12('Loose', 'Tight', 'Loose', 'Loose',
                          ['Yes','No'], 'Tight', 'Limited', 'Yes',
                          labels = ['Drop ON', 'Drop OFF'],
                          folder='plots_oct12/Drop/TightHAD0',
                          show=show)

    compare_filters_oct12('Loose', 'Loose', 'Loose', 'Loose',
                          ['Yes','No'], 'Tight', 'Limited', 'Yes',
                          labels = ['Drop ON', 'Drop OFF'],
                          folder='plots_oct12/Drop/LooseHAD0',
                          show=show)

ylabeldict = {
    'matchReco' : 'Matching Efficiency',
    'matchGen' : 'Reconstruction Efficiency'
}

pidnamedict = {
    'HAD0' : 'Neutral Hadron',
    'EM0' : 'Photon',
    'ELE' : 'Electron',
    'MU' : 'Muon',
    'HADCH' : 'Charged Hadron',
    'HADCH_HADCH' : 'Charged hadrons reconstructed as charged hadrons',
    'HADCH_MU' : "Muons reconstructed as charged hadrons",
    'HADCH_ELE' : "Electrons reconstructed as charged hadrons",
    'ELE_ELE' : 'Electrons reconstructed as electrons',
    'ELE_HADCH' : 'Charged hadrons reconstructed as electrons',
    'ELE_MU' : 'Muons reconstructed as electrons',
    'MU_HADCH' : 'Charged hadrons reconstructed as muons',
    'MU_ELE' : 'Electrons reconstructed as muons',
    'MU_MU' : 'Muons reconstructed as muons',
    'HAD0_HADCH' : 'Charged hadrons reconstructed as neutral hadrons',
    'HAD0_HAD0' : 'Neutral hadrons reconstructed as neutral hadrons',
    'EM0_ELE' : 'Electrons reconstructed as photons',
    'EM0_EM0' : "Photons reconstructed as photons"
}

def compare_filters_oct14(chfilters,
                          caloshares,
                          recoveries,
                          labels, 
                          folder=None, show=True, 
                          pids=['ELE', 'MU', 'HAD0', 'EM0', 'HADCH']):
    if type(chfilters) not in [list, tuple]:
        chfilters = [chfilters]
    if type(caloshares) not in [list, tuple]:
        caloshares = [caloshares]
    if type(recoveries) not in [list, tuple]:
        recoveries = [recoveries]

    maxlen = max(len(chfilters), len(caloshares), len(recoveries))
    if len(chfilters)==1:
        chfilters = chfilters*maxlen
    if len(caloshares)==1:
        caloshares = caloshares*maxlen
    if len(recoveries)==1:
        recoveries = recoveries*maxlen

    D = AOD

    etabins = [0,1,2]
    titles = ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']
    for etabin, title in zip(etabins, titles):
        for pid in pids:
            if '_' not in pid:
                for histname in ['matchReco', 'matchGen']:
                    plt.title("%s %s\n%s"%(pidnamedict[pid], 
                                           ylabeldict[histname], 
                                           title))
                    for i in range(maxlen):
                        label = labels[i]
                        chfilter = chfilters[i]
                        caloshare = caloshares[i]
                        recovery = recoveries[i]

                        fullname = '%s%s%s'%(chfilter, caloshare, recovery)
                        print(fullname)
                        plotting.plotMatch.plotMatchRate(D[fullname][histname], 'pt', 
                                pdgid=piddict[pid], etabin=etabin, 
                                label=label, show=False,
                                ylabel = ylabeldict[histname])
                    if folder is not None:
                        plt.savefig('%s/%s_eta%d_compareFilters_%s.png'%(folder,pid,etabin,histname), format='png', bbox_inches='tight')
                    if show:
                        plt.show()
                    else:
                        plt.clf()

            for ptbinmin, ptbinmax in zip([0, 2], [1, 25]):
                testname = '%s%s%s'%(chfilters[0], 
                                     caloshares[0], 
                                     recoveries[0])
                ptminval = D[testname]['HADCH']['dpt'].axes['pt'].edges[ptbinmin]
                ptmaxval = D[testname]['HADCH']['dpt'].axes['pt'].edges[ptbinmax]
                #plt.title(pid + " " + title + " " + "dpt\n %d < pt < %d GeV"%(ptminval, ptmaxval))
                plt.title("%s %s\n%s; %d < pt [GeV] < %d"%(
                                   pidnamedict[pid], 
                                   "",
                                   title,
                                   ptminval,
                                   ptmaxval))
                for i in range(maxlen):
                    label = labels[i]
                    chfilter = chfilters[i]
                    caloshare = caloshares[i]
                    recovery = recoveries[i]

                    fullname = '%s%s%s'%(chfilter, caloshare, recovery)

                    post.resolution.hist(D[fullname][pid]['dpt'], 'dpt',
                            etabin=etabin, density=False, fit=False,
                            label=label, show=False,
                            ptbinmin=ptbinmin, ptbinmax=ptbinmax)
                if folder is not None:
                    plt.savefig('%s/%s_eta%d_compareFilters_pt%dto%d_dpt.png'%(folder,pid,etabin, ptbinmin, ptbinmax), format='png', bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.clf()

def study_chargedfilters_oct14(show=True):
    compare_filters_oct14(['Tight', 'Loose', 'Looser'],
                          'Off',
                          'No',
                          labels = ['Strict flavor matching',
                                    'Allow electron/hadron mid-ID',
                                    'Allow any charged mis-ID'],
                          pids = ['HADCH', 'HADCH_ELE', 'HADCH_MU',
                                  'ELE', 'ELE_HADCH', 'ELE_MU',
                                  'MU', 'MU_HADCH', 'MU_ELE'],
                          folder = 'plots_oct14/Charged',
                          show=show)

def study_caloshare_oct14(show=True):
    compare_filters_oct14('Looser',
                          ['Off', 'Normal', 'Thresholded', 'GenThresholded', 'Both'],
                          'No',
                          labels = ['No sharing',
                                    'Allow HAD0 in ECAL',
                                    'Allow HAD0 in hard ECAL',
                                    'Allow hard HAD0 in ECAL',
                                    'Allow two-way sharing'],
                          pids = ['HAD0', 'HAD0_EM0',
                                  'EM0', 'EM0_HAD0'],
                          folder = 'plots_oct14/CaloShare',
                          show=show)

def study_recover_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          ['Free', 'Limited', 'No'],
                          labels = ['Track recovery',
                                    'Track recovery w/ threshold',
                                    'No track recovery'],
                          pids = ['HAD0', 'HAD0_HADCH',
                                  'EM0', 'EM0_HADCH', 'EM0_ELE',
                                  'ELE', 'HADCH', 'MU'],
                          folder = 'plots_oct14/CaloShare',
                          show=show)

def jecplots_oct14(show=True):
    H = AOD['LooserThresholdedLimited']['jets']

    for etabin, title in zip([0,1,2], ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']):
        for ptmin, ptmax in zip([0, 4], [2, 10]):
            ptminval = H['jecRatio'].axes['pt'].edges[ptmin]
            ptmaxval = H['jecRatio'].axes['pt'].edges[ptmax]

            plt.title(title + "\n %d < pT [GeV] < %d"%(ptminval, ptmaxval))
            for term, label in zip(['responseTerm', 'puTerm', 'unrecoTerm', 'jecPred'],
                                   ['Energy response', 'Pileup', 'Missed gen', 'Total']):
                post.resolution.hist(H[term], term, label=label,
                                     etabin=etabin, fit=False, show=False,
                                     ptbinmin=ptmin, ptbinmax=ptmax)
            plt.savefig('plots_oct14/JEC/jec_eta%d_pt%dto%d.png'%(etabin, ptmin, ptmax), format='png', bbox_inches='tight')
            plt.show()

def hadchplots_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['HADCH_HADCH', 'HADCH_ELE', 'HADCH_MU'],
                          folder = 'plots_oct14/HADCH/res',
                          show=show)

    compare_filters_oct14(['Tight', 'Looser', 'Looser'],
                          'Thresholded',
                          ['No', 'No', 'Limited'],
                          labels = ['Match with only charged hadrons',
                                    'Match with any charged',
                                    'Match with any charged + track recovery'],
                          pids = ['HADCH'],
                          folder = 'plots_oct14/HADCH/matchGen',
                          show=show)

    compare_filters_oct14(['Tight', 'Looser'],
                          'Thresholded',
                          ['No', 'No'],
                          labels = ['Match with only charged hadrons',
                                    'Match with any charged'],
                          pids = ['HADCH'],
                          folder = 'plots_oct14/HADCH/matchReco',
                          show=show)

def eleplots_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['ELE_ELE', 'ELE_HADCH', 'ELE_MU'],
                          folder = 'plots_oct14/ELE/res',
                          show=show)

    compare_filters_oct14(['Tight', 'Looser', 'Looser'],
                          'Thresholded',
                          ['No', 'No', 'Limited'],
                          labels = ['Match with only electrons',
                                    'Match with any charged',
                                    'Match with any charged + track recovery'],
                          pids = ['ELE'],
                          folder = 'plots_oct14/ELE/matchGen',
                          show=show)

    compare_filters_oct14(['Tight', 'Looser'],
                          'Thresholded',
                          ['No', 'No'],
                          labels = ['Match with only electrons',
                                    'Match with any charged'],
                          pids = ['ELE'],
                          folder = 'plots_oct14/ELE/matchReco',
                          show=show)

def muplots_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['MU_MU', 'MU_ELE', 'MU_HADCH'],
                          folder = 'plots_oct14/MU/res',
                          show=show)

    compare_filters_oct14(['Tight', 'Looser'],
                          'Thresholded',
                          ['No', 'No'],
                          labels = ['Match with only muons',
                                    'Match with any charged'],
                          pids = ['MU'],
                          folder = 'plots_oct14/MU/matchGen',
                          show=show)

def photonplots_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['EM0', 'EM0_ELE'],
                          folder = 'plots_oct14/EM0/res',
                          show=show)

    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['EM0'],
                          folder = 'plots_oct14/EM0/matchGen',
                          show=show)

    compare_filters_oct14('Looser',
                          'Thresholded',
                          ['No', 'Limited'],
                          labels = ['No track recovery',
                                    'Track recovery'],
                          pids = ['EM0'],
                          folder = 'plots_oct14/EM0/matchReco',
                          show=show)

def had0plots_oct14(show=True):
    compare_filters_oct14('Looser',
                          'Thresholded',
                          'Limited',
                          labels = [None],
                          pids = ['HAD0_HAD0', 'HAD0_HADCH'],
                          folder = 'plots_oct14/HAD0/res',
                          show=show)

    compare_filters_oct14('Looser',
                          ['Off', 'Thresholded'],
                          ['No', 'No'],
                          labels = ['Match only with neutral hadrons',
                                    'Match with any neutrals'],
                          pids = ['HAD0'],
                          folder = 'plots_oct14/HAD0/matchGen',
                          show=show)

    compare_filters_oct14('Looser',
                          'Thresholded',
                          ['No', 'Limited'],
                          labels = ['No track recovery',
                                    'Track recovery'],
                          pids = ['HAD0'],
                          folder = 'plots_oct14/HAD0/matchReco',
                          show=show)

def dRscan_oct15(folder=None, show=True):
    cuts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    etabins = [0,1,2]
    titles = ['Barrel', 'Barrel/Endcap Transition', 'Endcaps']
    pids = ['EM0', 'HAD0', 'HADCH', 'ELE', 'MU']
    histnames = ['matchReco', 'matchGen']

    etapairs = zip(etabins, titles)

    for setting in itertools.product(etapairs, pids, histnames):
        (etabin, title), pid, histname = setting

        pdgid = piddict[pid]

        plt.title('%s %s\n%s'%(pidnamedict[pid], ylabeldict[histname],title))
        for cut in cuts:
            name = 'dRx%d'%cut
            plotting.plotMatch.plotMatchRate(AOD[name][histname], 'pt', 
                                             etabin=etabin, label='%02d'%cut,
                                             pdgid=pdgid, show=False,
                                             ylabel = ylabeldict[histname])
        if folder is not None:
            plt.savefig("%s/%s_eta%d_%s_dRscan.png"%(folder,pid,
                                                     etabin,histname), 
                        format='png', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()

