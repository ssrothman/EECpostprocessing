import numpy as np
from time import time
import hist
import awkward as ak

import pyarrow.dataset as ds
import pyarrow.compute as pc

from tqdm import tqdm

from util import pa_mod, fill_bootstrapped

ptbins = [50, 88, 150, 254, 408, 1500]
Rbins = [0.3, 0.4, 0.5, 0.6]
rbins = np.linspace(0, 1, 16)
cbins = np.linspace(0, np.pi/2, 16)

prebinned_bins = {
    'R' : [0.0, 0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'r' : [0.0  , 0.025, 0.05 , 0.075, 
           0.1  , 0.125, 0.15 , 0.175, 0.2  ,
           0.225, 0.25 , 0.275, 0.3  , 0.325,
           0.35 , 0.375, 0.4  , 0.425,
           0.45 , 0.475, 0.5  , 0.525, 0.55 , 
           0.575, 0.6  , 0.625, 0.65 ,
           0.675, 0.7  , 0.725, 0.75 , 0.775, 
           0.8  , 0.825, 0.85 , 0.875,
           0.9  , 0.925, 0.95 , 0.975, 1.0  ],
    'c' : [0.0       , 0.10471976, 0.20943951, 
           0.31415927, 0.41887902,
           0.52359878, 0.62831853, 0.73303829, 
           0.83775804, 0.9424778 ,
           1.04719755, 1.15191731, 1.25663706, 
           1.36135682, 1.46607657, 1.57079633]
}

def fill_hist_from_parquet(basepath, Nboot, systwt, 
                           rng_offset, r123type,
                           prebinned, nbatch, skipNominal,
                           statN, statK,
                           kinreweight,
                           fs,
                           collect_debug_info):

    dataset = ds.dataset(basepath, format="parquet", filesystem=fs)

    the_evtwt = 'evtwt_%s'%systwt

    if skipNominal: 
        minboot = 1
    else:
        minboot = 0

    if prebinned:
        H_target = hist.Hist(
            hist.axis.Integer(minboot, Nboot+1,
                              name='bootstrap', label='bootstrap',
                              underflow=False, overflow=False),
            hist.axis.Variable(ptbins,
                               name='pt', label='$p_{T}$ [GeV]',
                               underflow=True, overflow=True),
            hist.axis.Variable(prebinned_bins['R'],
                              name="R", label = '$R$',
                              underflow=True, overflow=True),
            hist.axis.Variable(prebinned_bins['r'],
                              name="r", label = '$r$',
                              underflow=False, overflow=False),
            hist.axis.Variable(prebinned_bins['c'],
                              name="c", label = '$c$',
                              underflow=False, overflow=False),
            storage=hist.storage.Double()
        )
        H = hist.Hist(
            hist.axis.Integer(minboot, Nboot+1,
                              name='bootstrap', label='bootstrap',
                              underflow=False, overflow=False),
            hist.axis.Variable(ptbins,
                               name='pt', label='$p_{T}$ [GeV]',
                               underflow=True, overflow=True),
            hist.axis.Integer(1, len(prebinned_bins['R']),
                              name="R", label = '$R$',
                              underflow=True, overflow=True),
            hist.axis.Integer(1, len(prebinned_bins['r']),
                              name="r", label = '$r$',
                              underflow=False, overflow=False),
            hist.axis.Integer(1, len(prebinned_bins['c']),
                              name="c", label = '$c$',
                              underflow=False, overflow=False),
            storage=hist.storage.Double()
        )
    else:
        H = hist.Hist(
            hist.axis.Integer(minboot, Nboot+1,
                              name='bootstrap', label='bootstrap',
                              underflow=False, overflow=False),
            hist.axis.Variable(ptbins,
                               name='pt', label='$p_{T}$ [GeV]',
                               underflow=True, overflow=True),
            hist.axis.Variable(Rbins,
                              name="R", label = '$R$',
                              underflow=True, overflow=True),
            hist.axis.Variable(rbins,
                              name="r", label = '$r$',
                              underflow=False, overflow=False),
            hist.axis.Variable(cbins,
                              name="c", label = '$c$',
                              underflow=False, overflow=False),
            storage=hist.storage.Double()
        )


    if nbatch > 0:
        ibatch = 0

    time_tonpy = 0
    time_fill = 0

    if statN > 0:
        thefilter = pa_mod(ds.field('event'), statN) == statK
    else:
        thefilter = None

    if kinreweight is not None:
        reweightvars = ['Zpt', 'Zy']
    else:
        reweightvars = []

    iterator = tqdm(dataset.to_batches(columns=['R', 'r', 'c', 
                                                  'pt', 'wt', 
                                                  the_evtwt, 
                                                'event',
                                                'run',
                                                'lumi',
                                                *reweightvars],
                                         batch_readahead=2,
                                         fragment_readahead=2,
                                         use_threads=False,
                                         batch_size = 1<<20,
                                         filter = thefilter))

    total_rows = dataset.count_rows()
    rows_so_far = 0
    
    if collect_debug_info:
        all_counters = np.empty(0, dtype=np.uint64) 
        if r123type == 'philox':
            all_keys = np.empty(0, dtype=np.uint32)
        elif r123type == 'threefry':
            all_keys = np.empty(0, dtype=np.uint64)

        all_poisson1 = np.empty((Nboot, 0), dtype=np.uint64)

    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        if nbatch > 0:
            if ibatch >= nbatch:
                break
            ibatch += 1

        t0 = time()
        filldict = {}

        filldict['R'] = batch['R'].to_numpy(zero_copy_only=False)
        filldict['r'] = batch['r'].to_numpy(zero_copy_only=False)
        filldict['c'] = batch['c'].to_numpy(zero_copy_only=False)
        filldict['pt'] = batch['pt'].to_numpy(zero_copy_only=False)

        weight = batch[the_evtwt].to_numpy(zero_copy_only=False) 
        weight = weight * batch['wt'].to_numpy(zero_copy_only=False)

        if kinreweight is not None:
            Zpt = batch['Zpt'].to_numpy(zero_copy_only=False)
            Zy = batch['Zy'].to_numpy(zero_copy_only=False)
            kinwt = kinreweight(Zpt, Zy)
            weight = weight * kinwt

        time_tonpy += time() - t0

        if prebinned:
            if np.min(r) < 1:
                print("WARNING: r < 1")
            if np.min(c) < 1:
                print("WARNING: c < 1")

            if np.max(r) >= len(prebinned_bins['r']):
                print("WARNING: r >= %d" % len(prebinned_bins['r']))
            if np.max(c) >= len(prebinned_bins['c']):
                print("WARNING: c >= %d" % len(prebinned_bins['c']))

        t0 = time()
        debug = fill_bootstrapped(H, filldict, weight, 
                                  r123type, Nboot, rng_offset, batch, 
                                  skipNominal, collect_debug_info)
        time_fill += time() - t0

        if collect_debug_info:
            boots, repeats, counters, keys = debug
            all_poisson1 = np.concatenate((all_poisson1, boots), axis=1)
            all_counters = np.concatenate((all_counters, counters))
            all_keys = np.concatenate((all_keys, keys))

    print("Time to numpy: %.2f" % time_tonpy)
    print("Time to fill: %.2f" % time_fill)
    print("Total time: %.2f" % (time_fill + time_tonpy))

    if prebinned:
        H_target += H.values(flow=True)
        return H_target

    if collect_debug_info: 
        return H, all_poisson1, all_counters, all_keys
    else:
        return H

def fill_transferhist_from_parquet(basepath, Nboot, systwt,
                                   rng_offset, skipNominal,
                                   r123type,
                                   statN, statK,
                                   kinreweight,
                                   fs):

    dataset = ds.dataset(basepath, format="parquet", filesystem=fs)

    the_evtwt = 'evtwt_%s'%systwt
    
    if skipNominal:
        minboot = 1
    else:
        minboot = 0


    H = hist.Hist(
        hist.axis.Integer(minboot, Nboot+1, 
                          name='bootstrap', label='bootstrap',
                          underflow=False, overflow=False),
        hist.axis.Variable(ptbins,
                           name='pt_reco', label='$p_{T,reco}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Variable(Rbins,
                          name="R_reco", label = '$R_{reco}$',
                          underflow=True, overflow=True),
        hist.axis.Variable(rbins,
                          name="r_reco", label = '$r_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Variable(cbins,
                          name="c_reco", label = '$c_{reco}$',
                          underflow=False, overflow=False),
        hist.axis.Variable(ptbins,
                           name='pt_gen', label='$p_{T,gen}$ [GeV]',
                           underflow=True, overflow=True),
        hist.axis.Variable(Rbins,
                          name="R_gen", label = '$R_{gen}$',
                          underflow=True, overflow=True),
        hist.axis.Variable(rbins,
                          name="r_gen", label = '$r_{gen}$',
                          underflow=False, overflow=False),
        hist.axis.Variable(cbins,
                          name="c_gen", label = '$c_{gen}$',
                          underflow=False, overflow=False),
        storage=hist.storage.Double()
    )


    if statN > 0:
        thefilter = pa_mod(ds.field('event'), statN) == statK
    else:
        thefilter = None

    if kinreweight is not None:
        reweightvars = ['Zpt', 'Zy']
    else:
        reweightvars = []

    for batch in tqdm(dataset.to_batches(columns=['pt_reco', 'R_reco', 
                                                  'r_reco', 'c_reco',
                                                  'pt_gen', 'R_gen',
                                                  'r_gen', 'c_gen',
                                                  'wt_reco', 'wt_gen',
                                                  'event', the_evtwt,
                                                  'run', 'lumi',
                                                  *reweightvars],
                                         batch_readahead=2,
                                         fragment_readahead=2,
                                         use_threads=False,
                                         batch_size = 1<<20,
                                         filter = thefilter)):
        
        filldict = {}
        filldict['R_reco']   =batch['R_reco'].to_numpy(zero_copy_only=False)
        filldict['R_gen']    =batch['R_gen'].to_numpy(zero_copy_only=False)
        filldict['r_reco']   =batch['r_reco'].to_numpy(zero_copy_only=False)
        filldict['r_gen']    =batch['r_gen'].to_numpy(zero_copy_only=False)
        filldict['c_reco']   =batch['c_reco'].to_numpy(zero_copy_only=False)
        filldict['c_gen']    =batch['c_gen'].to_numpy(zero_copy_only=False)
        filldict['pt_reco']  =batch['pt_reco'].to_numpy(zero_copy_only=False)
        filldict['pt_gen']   =batch['pt_gen'].to_numpy(zero_copy_only=False)

        weight=batch[the_evtwt].to_numpy(zero_copy_only=False)
        weight=weight*batch['wt_reco'].to_numpy(zero_copy_only=False)

        if kinreweight is not None:
            Zpt = batch['Zpt'].to_numpy(zero_copy_only=False)
            Zy = batch['Zy'].to_numpy(zero_copy_only=False)
            kinwt = kinreweight(Zpt, Zy)
            weight = weight * kinwt


        fill_bootstrapped(H, filldict, weight, 
                          r123type, Nboot, rng_offset, batch,
                          skipNominal)

    return H
