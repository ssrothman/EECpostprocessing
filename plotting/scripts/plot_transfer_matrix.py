#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

from unfolding.detectormodel import DetectorModel

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='/eos/user/d/dponman/proj_unfold_workspace')
parser.add_argument('--output',    type=str, default='plots/proj')
args = parser.parse_args()

JPT_BINS = [
    (40,   100),
    (100,  200),
    (200,  340),
    (340,  520),
    (520,  740),
    (740,  1000),
    (1000, 1500),
    (1500, 2500),
    (2500, 5000),
]

R_EDGES = np.array([
    0.001, 0.001193, 0.001423, 0.001697, 0.002024, 0.002414, 0.00288, 0.003435,
    0.004097, 0.004886, 0.005828, 0.006952, 0.008292, 0.00989, 0.011797, 0.014071,
    0.016783, 0.020019, 0.023878, 0.02848, 0.033971, 0.040519, 0.04833, 0.057646,
    0.068758, 0.082012, 0.097821, 0.116678, 0.139169, 0.165997, 0.197995, 0.236162,
    0.281685, 0.335985, 0.400751, 0.478002, 0.570145, 0.680049, 0.811139, 0.967499,
    1.153999, 1.376451, 1.641783, 1.958262, 2.335748, 2.786, 3.323046, 3.963615,
    4.727664, 5.638995, 6.726
])

valid = np.load(os.path.join(args.workspace, 'valid_bins.npy'))
model = DetectorModel.from_disk(os.path.join(args.workspace, 'detectormodel'))

transfer = model._transfer0  # (nReco, nGen)

n_first = int(valid.sum()) - 8 * 50
r_start = 50 - n_first

jpt_slices = []
idx = 0
for i in range(len(JPT_BINS)):
    if i == 0:
        jpt_slices.append(slice(idx, idx + n_first))
        idx += n_first
    else:
        jpt_slices.append(slice(idx + r_start, idx + 50))
        idx += 50

r_edges = R_EDGES[r_start:]
r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])

os.makedirs(args.output, exist_ok=True)
hep.style.use('CMS')

for i, (jlo, jhi) in enumerate(JPT_BINS):
    sl = jpt_slices[i]
    idx_arr = np.arange(sl.start, sl.stop)
    T = transfer[np.ix_(idx_arr, idx_arr)]  # reco x gen block for this Jpt bin

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(r_centers, r_centers, T, cmap='Blues')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Gen $\Delta R$', fontsize=11)
    ax.set_ylabel(r'Reco $\Delta R$', fontsize=11)
    fig.colorbar(mesh, ax=ax, label='Transfer probability')
    hep.cms.label(ax=ax, data=False, text='Private', com=13)
    ax.set_title(f'Jpt {jlo}–{jhi} GeV', fontsize=11)

    fig.tight_layout()
    fname = os.path.join(args.output, f'transfer_Jpt{jlo}-{jhi}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print('Saved', fname)
