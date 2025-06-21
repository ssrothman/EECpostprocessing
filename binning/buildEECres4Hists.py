import numpy as np
from time import time
import hist
import awkward as ak

import pyarrow.dataset as ds
import pyarrow.compute as pc

from tqdm import tqdm

from pyrandom123 import get_poisson1_philox2x32

ptbins = [50, 88, 150, 254, 408, 1500]
Rbins = [0.3, 0.4, 0.5, 0.6]
#ptbins = [150, 300, 600, 1200, 2400]
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

def pa_mod(val, divisor):
    quotient = pc.floor(pc.divide(val, divisor))
    remainder = pc.subtract(val, pc.multiply(quotient, divisor))
    return remainder

def fill_hist_from_parquet(basepath, bootstrap, systwt, random_seed=0,
                           prebinned = False, nbatch=-1, skipNominal=False,
                           statN=-1, statK=-1,
                           kinreweight=None,
                           fs=None):

    dataset = ds.dataset(basepath, format="parquet", filesystem=fs)

    the_evtwt = 'evtwt_%s'%systwt

    if skipNominal: 
        minboot = 1
    else:
        minboot = 0

    if prebinned:
        H_target = hist.Hist(
            hist.axis.Integer(minboot, bootstrap+1,
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
            hist.axis.Integer(minboot, bootstrap+1,
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
            hist.axis.Integer(minboot, bootstrap+1,
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
    time_fillnom = 0
    time_fillboot = 0
    time_bootwt = 0

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
    
    #all_poisson = np.zeros((bootstrap, 0), dtype=np.uint32)
    #all_keys = np.zeros((bootstrap, 0), dtype=np.uint32)
    #all_counters = np.zeros((bootstrap, 0), dtype=np.uint64)
    #edges = []

    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        if nbatch > 0:
            if ibatch >= nbatch:
                break
            ibatch += 1

        t0 = time()
        R = batch['R'].to_numpy(zero_copy_only=False)
        r = batch['r'].to_numpy(zero_copy_only=False)
        c = batch['c'].to_numpy(zero_copy_only=False)

        pt = batch['pt'].to_numpy(zero_copy_only=False)
        evtwt = batch[the_evtwt].to_numpy(zero_copy_only=False)
        wt = batch['wt'].to_numpy(zero_copy_only=False)

        if kinreweight is not None:
            Zpt = batch['Zpt'].to_numpy(zero_copy_only=False)
            Zy = batch['Zy'].to_numpy(zero_copy_only=False)
            kinwt = kinreweight(Zpt, Zy)
            evtwt = evtwt * kinwt

        run = batch['run'].to_numpy(zero_copy_only=False).astype(np.uint64)
        lumi = batch['lumi'].to_numpy(zero_copy_only=False).astype(np.uint64)
        event = batch['event'].to_numpy(zero_copy_only=False).astype(np.uint64) 
        evthash = ((run << 48) + (lumi << 32) + event).astype(np.uint64);

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

        if not skipNominal:
            t0 = time()
            H.fill(
                R=R,
                r=r,
                c=c,
                pt=pt,
                bootstrap = 0,
                weight=evtwt*wt,
            )
            time_fillnom += time() - t0

        t0 = time()
        repeats = ak.to_numpy(ak.run_lengths(evthash))
        cumsum = np.cumsum(repeats) - 1
        unique_hashes = evthash[cumsum]

        keys = (np.arange(bootstrap, dtype=np.uint32) + np.uint32(random_seed*bootstrap))[:, None]
        counters = unique_hashes[None, :].astype(np.uint64)

        boots = get_poisson1_philox2x32(counters, keys)

        #all_poisson = np.concatenate((all_poisson, boots), axis=1)
        #keys_b, counters_b = np.broadcast_arrays(keys, counters)
        #all_keys = np.concatenate((all_keys, keys_b), axis=1)
        #all_counters = np.concatenate((all_counters, counters_b), axis=1)
        #edges.append(all_counters.shape[1])

        time_bootwt += time() - t0

        #TEST
        #print()
        #print("counters", counters)
        #print("\t", counters.dtype, counters.shape)
        #print("keys", keys)
        #print("\t", keys.dtype, keys.shape)
        #print()
    
        for i in range(bootstrap):
            t0 = time()
            H.fill(
                R=R,
                r=r,
                c=c,
                pt=pt,
                bootstrap = i+1,
                weight=evtwt*wt*np.repeat(boots[i], repeats),
            )
            time_fillboot += time() - t0


    print("Time to numpy: %.2f" % time_tonpy)
    print("Time to fill nominal: %.2f" % time_fillnom)
    print("Time to fill bootstrap: %.2f" % time_fillboot)
    print("Time to bootstrap weight: %.2f" % time_bootwt)
    print("Total time: %.2f" % (time_fillnom + time_fillboot + time_tonpy))

    if prebinned:
        H_target += H.values(flow=True)
        return H_target

    return H, None #, all_poisson, all_keys, all_counters, edges

def fill_transferhist_from_parquet(basepath, bootstrap, systwt,
                                   random_seed = 0, skipNominal=False,
                                   statN=-1, statK=-1,
                                   kinreweight=None,
                                   fs=None):

    dataset = ds.dataset(basepath, format="parquet", filesystem=fs)

    the_evtwt = 'evtwt_%s'%systwt
    
    if skipNominal:
        minboot = 1
    else:
        minboot = 0


    H = hist.Hist(
        hist.axis.Integer(minboot, bootstrap+1, 
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
        
        R_reco=batch['R_reco'].to_numpy(zero_copy_only=False)
        R_gen=batch['R_gen'].to_numpy(zero_copy_only=False)
        r_reco=batch['r_reco'].to_numpy(zero_copy_only=False)
        r_gen=batch['r_gen'].to_numpy(zero_copy_only=False)
        c_reco=batch['c_reco'].to_numpy(zero_copy_only=False)
        c_gen=batch['c_gen'].to_numpy(zero_copy_only=False)
        pt_reco=batch['pt_reco'].to_numpy(zero_copy_only=False)
        pt_gen=batch['pt_gen'].to_numpy(zero_copy_only=False)
        weight=batch[the_evtwt].to_numpy(zero_copy_only=False)*batch['wt_reco'].to_numpy(zero_copy_only=False)

        if kinreweight is not None:
            Zpt = batch['Zpt'].to_numpy(zero_copy_only=False)
            Zy = batch['Zy'].to_numpy(zero_copy_only=False)
            kinwt = kinreweight(Zpt, Zy)
            weight = weight * kinwt

        if not skipNominal:
            H.fill(
                R_reco  = R_reco,
                R_gen   = R_gen,
                r_reco  = r_reco,
                r_gen   = r_gen,
                c_reco  = c_reco,
                c_gen   = c_gen,
                pt_reco = pt_reco,
                pt_gen  = pt_gen,
                weight  = weight,
                bootstrap = 0
            )
        

        run = batch['run'].to_numpy(zero_copy_only=False).astype(np.uint64)
        lumi = batch['lumi'].to_numpy(zero_copy_only=False).astype(np.uint64)
        event = batch['event'].to_numpy(zero_copy_only=False).astype(np.uint64) 
        evthash = ((run << 32) + (lumi << 16) + event).astype(np.uint64);

        repeats = ak.to_numpy(ak.run_lengths(evthash))
        cumsum = np.cumsum(repeats) - 1
        unique_hashes = evthash[cumsum]

        keys = (np.arange(bootstrap, dtype=np.uint32) + np.uint32(random_seed*bootstrap))[:, None]
        counters = unique_hashes[None, :].astype(np.uint64)

        boots = get_poisson1_philox2x32(counters, keys)

        for i in range(bootstrap):
            boot = boots[i]
            boot = np.repeat(boot, repeats)
            H.fill(
                R_reco  = R_reco,
                R_gen   = R_gen,
                r_reco  = r_reco,
                r_gen   = r_gen,
                c_reco  = c_reco,
                c_gen   = c_gen,
                pt_reco = pt_reco,
                pt_gen  = pt_gen,
                weight  = weight*boot,
                bootstrap = i+1
            )

    return H
