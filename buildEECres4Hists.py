import numpy as np
from time import time
import hist
import awkward as ak

import pyarrow.dataset as ds

from tqdm import tqdm

ptbins = [50, 88, 150, 254, 408, 1500]
Rbins = [0.3, 0.4, 0.5, 0.6]
#ptbins = [150, 300, 600, 1200, 2400]
rbins = np.linspace(0, 1, 16)
cbins = np.linspace(0, np.pi/2, 16)

class sm64_rng:
    def __init__(self, seedarr, repeats):
        self.repeats = repeats
        cumsum = np.cumsum(self.repeats) - 1
        self.state = seedarr[cumsum] # only unique values

    def next_uint64(self):
        self.state += 0x9e3779b97f4a7c15;
        z = self.state
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31))
        return z

    def next_uint64_masked(self, mask):
        self.state[mask] += 0x9e3779b97f4a7c15
        z = self.state[mask]
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        z = (z ^ (z >> 31))
        return z

    def next_float(self):
        return self.next_uint64().astype(np.float64) / (1<<64)

    def next_float_masked(self, mask):
        return self.next_uint64_masked(mask) / (1<<64)

    def next_poisson(self, lam):
        # Poisson RNG using the rejection method
        if lam < 0:
            raise ValueError("Lambda must be non-negative")
        elif lam == 0:
            return 0

        L = np.exp(-lam)
        p = self.next_float()
        k = np.ones_like(p, dtype=np.int64)
        while True:
            mask = p > L
            if (not np.any(mask)):
                break
            p[mask] = p[mask] * self.next_float_masked(mask)
            k[mask] += 1
        return np.repeat(k - 1, self.repeats) #unpack the run lengths

def fill_hist_from_parquet(basepath, bootstrap, systwt, random_seed=0):

    dataset = ds.dataset(basepath, format="parquet")

    the_evtwt = 'evtwt_%s'%systwt

    H = hist.Hist(
        hist.axis.Integer(0, bootstrap+1,
                          name='bootstrap', label='bootstrap',
                          underflow=True, overflow=True),
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

    for batch in tqdm(dataset.to_batches(columns=['R', 'r', 'c', 
                                                  'pt', 'wt', 
                                                  the_evtwt, 'eventhash'],
                                         batch_size=1000000000)):
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

        H.fill(
            R=R,
            r=r,
            c=c,
            pt=pt,
            bootstrap = 0,
            weight=evtwt*wt,
        )

        repeats = ak.to_numpy(ak.run_lengths(evthash))
        rng = sm64_rng(evthash + random_seed, repeats)

        for i in range(bootstrap):
            boot = rng.next_poisson(1.0)
            H.fill(
                R=R,
                r=r,
                c=c,
                pt=pt,
                bootstrap = i+1,
                weight=evtwt*wt*boot,
            )

    return H

def fill_transferhist_from_parquet(basepath, systwt):

    dataset = ds.dataset(basepath, format="parquet")

    the_evtwt = 'evtwt_%s'%systwt

    H = hist.Hist(
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

    for batch in tqdm(dataset.to_batches(columns=['pt_reco', 'R_reco', 
                                                  'r_reco', 'c_reco',
                                                  'pt_gen', 'R_gen',
                                                  'r_gen', 'c_gen',
                                                  'wt_reco', 'wt_gen',
                                                  the_evtwt])):
        
        H.fill(
            R_reco=batch['R_reco'].to_numpy(),
            R_gen=batch['R_gen'].to_numpy(),
            r_reco=batch['r_reco'].to_numpy(),
            r_gen=batch['r_gen'].to_numpy(),
            c_reco=batch['c_reco'].to_numpy(),
            c_gen=batch['c_gen'].to_numpy(),
            pt_reco=batch['pt_reco'].to_numpy(),
            pt_gen=batch['pt_gen'].to_numpy(),
            weight=batch[the_evtwt].to_numpy()*batch['wt_reco'].to_numpy(),
        )

    return H


def fill_wtratiohist_from_parquet(basepath, systwt):

    dataset = ds.dataset(basepath, format="parquet")

    the_evtwt = 'evtwt_%s'%systwt

    H_wtgen = hist.Hist(
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
    H_wtreco = hist.Hist(
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

    for batch in tqdm(dataset.to_batches(columns=['pt_reco', 'R_reco',
                                                  'r_reco', 'c_reco',
                                                  'pt_gen', 'R_gen',
                                                  'r_gen', 'c_gen',
                                                  'wt_reco', 'wt_gen',
                                                  the_evtwt])):
        H_wtgen.fill(
            R_gen=batch['R_gen'].to_numpy(),
            r_gen=batch['r_gen'].to_numpy(),
            c_gen=batch['c_gen'].to_numpy(),
            pt_gen=batch['pt_gen'].to_numpy(),
            weight=batch[the_evtwt].to_numpy()*batch['wt_gen'].to_numpy(),
        )
        H_wtreco.fill(
            R_gen=batch['R_gen'].to_numpy(),
            r_gen=batch['r_gen'].to_numpy(),
            c_gen=batch['c_gen'].to_numpy(),
            pt_gen=batch['pt_gen'].to_numpy(),
            weight=batch[the_evtwt].to_numpy()*batch['wt_reco'].to_numpy(),
        )

    Hresult = H_wtgen.copy().reset()
    Hresult += H_wtreco.values(flow=True)/H_wtgen.values(flow=True)

    return Hresult, H_wtgen, H_wtreco
