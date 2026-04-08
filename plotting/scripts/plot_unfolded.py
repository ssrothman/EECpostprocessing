#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='proj_unfold_workspace')
parser.add_argument('--output',    type=str, default='plots/proj')
args = parser.parse_args()

R_EDGES = np.array([
    0.001, 0.001193, 0.001423, 0.001697, 0.002024, 0.002414, 0.00288, 0.003435,
    0.004097, 0.004886, 0.005828, 0.006952, 0.008292, 0.00989, 0.011797, 0.014071,
    0.016783, 0.020019, 0.023878, 0.02848, 0.033971, 0.040519, 0.04833, 0.057646,
    0.068758, 0.082012, 0.097821, 0.116678, 0.139169, 0.165997, 0.197995, 0.236162,
    0.281685, 0.335985, 0.400751, 0.478002, 0.570145, 0.680049, 0.811139, 0.967499,
    1.153999, 1.376451, 1.641783, 1.958262, 2.335748, 2.786, 3.323046, 3.963615,
    4.727664, 5.638995, 6.726
])

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

x          = np.load(os.path.join(args.workspace, 'minimization', 'result', 'x.npy'))
gen_values = np.load(os.path.join(args.workspace, 'gen', 'values.npy'))
valid      = np.load(os.path.join(args.workspace, 'valid_bins.npy'))

unfolded = x * gen_values

# bin structure: Jpt slow index, R fast index (C-order)
# first Jpt bin may be missing leading R bins (removed as NaN)
n_first = int(valid.sum()) - 8 * 50
jpt_slices  = []
jpt_r_edges = []
idx = 0
for i in range(len(JPT_BINS)):
    if i == 0:
        n_r     = n_first
        r_edges = R_EDGES[50 - n_r:]
    else:
        n_r     = 50
        r_edges = R_EDGES
    jpt_slices.append(slice(idx, idx + n_r))
    jpt_r_edges.append(r_edges)
    idx += n_r

os.makedirs(args.output, exist_ok=True)
hep.style.use('CMS')

for i, (jlo, jhi) in enumerate(JPT_BINS):
    fig, ax = plt.subplots(figsize=(8, 8))

    sl       = jpt_slices[i]
    r_edges  = jpt_r_edges[i]
    unf_vals = unfolded[sl]
    gen_vals = gen_values[sl]

    ax.stairs(unf_vals, r_edges, label='Unfolded', color='black')
    ax.stairs(gen_vals, r_edges, label='MC Gen',   color='red', linestyle='--')

    ax.set_xscale('log')
    if i > 0:
        ax.set_yscale('log')

    ax.set_xlabel('R')
    ax.set_ylabel('EEC')
    ax.legend()

    hep.cms.label(ax=ax, data=False, text='Private Simulation', com=13)

    fig.tight_layout()
    fname = os.path.join(args.output, f'unfolded_Jpt{jlo}-{jhi}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print('Saved', fname)
