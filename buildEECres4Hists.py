import numpy as np
from time import time
import hist
import awkward as ak

import pyarrow.dataset as ds
import pyarrow.compute as pc

from tqdm import tqdm

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

class RNG:
    '''
    This implements a vectorized random nunmber generator, 
    not in the sense that the production of random numbers
    from a given seed is vectorized, but in the sense that
    this class generates a vector of random numbers from 
    a vector of seeds. Ie this is under the hood a vector of 
    random number /generators/.

    This is desirable because it allows us to generate random numbers
    deterministically from each event's (run, event, lumi) value. This
    is useful for bootstrapping :)

    The actually RNG implementation is the implementation from 
    Numerical Recipes 3rd edition page 342 (366 in the pdf).
    This is basically a convlution of two xors and a linear shift
    Should be more than good enough :)
    '''
    def __init__(self, seeds):
        self.N = len(seeds)

        self.v = np.ones(self.N, dtype=np.uint64)*4101842887655102017
        self.w = np.ones(self.N, dtype=np.uint64)

        self.u = np.bitwise_xor(seeds.astype(np.uint64), self.v)
        self.next_int64()

        self.v = self.u.copy();
        self.next_int64()

        self.w = self.v.copy()
        self.next_int64()

        # For some random number generators [not sure about this one]
        # if you initialize with simialar seeds (ie differing by +-1) you
        # get strongly correlated numbers for the first few calls
        # I often initialize with seeds that differ by only +-1 
        # I'm not sure if this is genuinely a problem
        # But out of an abundance of caution, we'll just throw away
        # the first 8 random numbers
        # the overhead from doing this is negligible anyway
        # is it's no cost for maybe a small gain :)
        for i in range(8):
            self.next_int64()

    def next_int64(self, mask=slice(None)):
        self.u[mask] = self.u[mask] * 2862933555777941757 + 7046029254386353087

        self.v[mask] = np.bitwise_xor(self.v[mask], self.v[mask] >> 17)
        self.v[mask] = np.bitwise_xor(self.v[mask], self.v[mask] << 31)
        self.v[mask] = np.bitwise_xor(self.v[mask], self.v[mask] >> 8)

        self.w[mask] = 4294957665 * (np.bitwise_and(self.w[mask], 0xffffffff)) + (self.w[mask] >> 32)

        x = np.bitwise_xor(self.u[mask], self.u[mask] << 21)
        x = np.bitwise_xor(x, x >> 35)
        x = np.bitwise_xor(x, x << 4)
        result = np.bitwise_xor(x + self.v[mask], self.w[mask])
        return result

    def next_int32(self):
        return self.next_int64() & 0xffffffff

    def next_float64(self, mask=slice(None)):
        # get uniform random number on range [0, 1) from unsigned int64
        # for reasons beyond my comprehension, I don't actually get 64 bits
        # from next_int64(), but rather 57 bits
        return self.next_int64(mask) / ((1 << 64) )

    def next_poisson(self, lamexp=np.exp(-1.0)):
        k = np.zeros(self.N, dtype=np.int32)
        t = self.next_float64()
        
        mask = t > lamexp
        while np.any(mask):
            k[mask] += 1
            t[mask] *= self.next_float64(mask)
            mask = t > lamexp

        return k

def fill_hist_from_parquet(basepath, bootstrap, systwt, random_seed=0,
                           prebinned = False, nbatch=-1, skipNominal=False,
                           statN=-1, statK=-1,
                           ptreweight_func=None,
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
        thefilter = pa_mod(ds.field('eventhash'), statN) == statK
    else:
        thefilter = None

    iterator = tqdm(dataset.to_batches(columns=['R', 'r', 'c', 
                                                  'pt', 'wt', 
                                                  the_evtwt, 'eventhash'],
                                         batch_readahead=2,
                                         fragment_readahead=2,
                                         use_threads=False,
                                         batch_size = 1<<20,
                                         filter = thefilter))

    total_rows = dataset.count_rows()
    rows_so_far = 0
    
    for batch in iterator:
        rows_so_far += batch.num_rows
        iterator.set_description("Rows: %g%%" % (rows_so_far / total_rows * 100))

        if nbatch > 0:
            if ibatch >= nbatch:
                break
            ibatch += 1

        t0 = time()
        R = batch['R'].to_numpy()
        r = batch['r'].to_numpy()
        try:
            c = batch['c'].to_numpy()
        except:
            print("error")
            print(c)
            c = batch['c'].to_numpy(zero_copy_only=False)

        pt = batch['pt'].to_numpy()
        evtwt = batch[the_evtwt].to_numpy()
        wt = batch['wt'].to_numpy()
        evthash = batch['eventhash'].to_numpy()
        time_tonpy += time() - t0

        if ptreweight_func is not None:
            wt = wt * ptreweight_func(pt)

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

        repeats = ak.to_numpy(ak.run_lengths(evthash))
        cumsum = np.cumsum(repeats) - 1
        seeds = evthash[cumsum] + random_seed
        rng = RNG(seeds)

        for i in range(bootstrap):
            t0 = time()
            boot = rng.next_poisson()
            boot = np.repeat(boot, repeats)
            time_bootwt += time() - t0
            t0 = time()
            H.fill(
                R=R,
                r=r,
                c=c,
                pt=pt,
                bootstrap = i+1,
                weight=evtwt*wt*boot,
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

    return H

def fill_transferhist_from_parquet(basepath, bootstrap, systwt,
                                   random_seed = 0, skipNominal=False,
                                   statN=-1, statK=-1,
                                   ptreweight_func=None,
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
        thefilter = pa_mod(ds.field('eventhash'), statN) == statK
    else:
        thefilter = None

    for batch in tqdm(dataset.to_batches(columns=['pt_reco', 'R_reco', 
                                                  'r_reco', 'c_reco',
                                                  'pt_gen', 'R_gen',
                                                  'r_gen', 'c_gen',
                                                  'wt_reco', 'wt_gen',
                                                  'eventhash', the_evtwt],
                                         batch_readahead=2,
                                         fragment_readahead=2,
                                         use_threads=False,
                                         batch_size = 1<<20,
                                         filter = thefilter)):
        
        R_reco=batch['R_reco'].to_numpy()
        R_gen=batch['R_gen'].to_numpy()
        r_reco=batch['r_reco'].to_numpy()
        r_gen=batch['r_gen'].to_numpy()
        c_reco=batch['c_reco'].to_numpy()
        c_gen=batch['c_gen'].to_numpy()
        pt_reco=batch['pt_reco'].to_numpy()
        pt_gen=batch['pt_gen'].to_numpy()
        weight=batch[the_evtwt].to_numpy()*batch['wt_reco'].to_numpy()

        if ptreweight_func is not None:
            weight = weight * ptreweight_func(pt_reco)

        evthash = batch['eventhash'].to_numpy()

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
        
        repeats = ak.to_numpy(ak.run_lengths(evthash))
        cumsum = np.cumsum(repeats) - 1
        seeds = evthash[cumsum] + random_seed
        rng = RNG(seeds)

        for i in range(bootstrap):
            boot = rng.next_poisson(1.0)
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
