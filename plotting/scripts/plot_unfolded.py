#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

from general.fslookup.skim_path import lookup_skim_path

parser = argparse.ArgumentParser()
parser.add_argument('--pythia-workspace', type=str, default='/eos/user/d/dponman/proj_unfold_workspace')
parser.add_argument('--data-workspace',   type=str, default='/eos/user/d/dponman/proj_unfold_workspace_data')
parser.add_argument('--herwig-workspace', type=str, default='/eos/user/d/dponman/proj_unfold_workspace_herwig')
parser.add_argument('--output',           type=str, default='plots/proj')
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

gen_values = np.load(os.path.join(args.pythia_workspace, 'gen', 'values.npy'))

x_pythia = np.load(os.path.join(args.pythia_workspace, 'minimization', 'result', 'x.npy'))
x_data   = np.load(os.path.join(args.data_workspace,   'minimization', 'result', 'x.npy'))
x_herwig = np.load(os.path.join(args.herwig_workspace, 'minimization', 'result', 'x.npy'))

data_reco   = np.load(os.path.join(args.data_workspace,   'reco', 'values.npy'))
herwig_reco = np.load(os.path.join(args.herwig_workspace, 'reco', 'values.npy'))
pythia_reco = np.load(os.path.join(args.pythia_workspace, 'reco', 'values.npy'))

pythia_ht_gen   = gen_values  # HT-stitched gen baseline from workspace
pythia_unfolded = x_pythia * gen_values
data_unfolded   = x_data   * gen_values
herwig_unfolded = x_herwig * gen_values
valid = np.load(os.path.join(args.pythia_workspace, 'valid_bins.npy'))

def load_gen(config_suite, runtag, dataset, objsyst='NOM', wtsyst='nominal'):
    fs, skimpath = lookup_skim_path(
        'dylan-lxplus-eos', config_suite, runtag, dataset, objsyst, 'proj_Gen'
    )
    with fs.open(skimpath + '_BINNED_%s.npy' % wtsyst, 'rb') as f:
        values = np.load(f)
    return values[50:-50][valid]

herwig_gen = load_gen('EvtMCprojConfig', 'herwig_v3', 'DYJetsToLL_Herwig')
valid_2d    = valid.reshape(len(JPT_BINS), 50)
jpt_slices  = []
jpt_r_edges = []
idx = 0
for i in range(len(JPT_BINS)):
    valid_r = np.where(valid_2d[i])[0]
    n_r     = len(valid_r)
    r_edges = np.append(R_EDGES[valid_r], R_EDGES[valid_r[-1] + 1])
    jpt_slices.append(slice(idx, idx + n_r))
    jpt_r_edges.append(r_edges)
    idx += n_r

def norm(vals, edges):
    dR = np.diff(edges)
    integral = np.sum(vals * dR)
    if integral == 0:
        return vals / dR
    return vals / (integral * dR)


os.makedirs(args.output, exist_ok=True)
hep.style.use('CMS')

C_PYTHIA_GEN  = '#3B2F2F'  # dark espresso
C_HERWIG_GEN  = '#59A14F'  # green
C_DATA_RECO   = '#2C2C2C'  # near black
C_DATA_UNF    = '#4E79A7'  # steel blue
C_HERWIG_UNF  = '#F28E2B'  # orange
C_PYTHIA_UNF  = '#B07AA1'  # purple
C_PYTHIA_RECO = '#E15759'  # red
C_HERWIG_RECO = '#76B7B2'  # tiel 

for i, (jlo, jhi) in enumerate(JPT_BINS):
    sl      = jpt_slices[i]
    r_edges = jpt_r_edges[i]

    curves = [
        ('Pythia Gen',      norm(pythia_ht_gen[sl],    r_edges), C_PYTHIA_GEN,  '-',  1.5),
#        ('Herwig Gen',      norm(herwig_gen[sl],       r_edges), C_HERWIG_GEN,  '-', 1.2),
#        ('Herwig Reco',     norm(herwig_reco[sl],      r_edges), C_HERWIG_RECO, '--', 1.0),
        ('Data Reco',       norm(data_reco[sl],        r_edges), C_DATA_RECO,   '--',  1.0),
        ('Data Unfolded',   norm(data_unfolded[sl],    r_edges), C_DATA_UNF,    ':',  1.0),
#        ('Herwig Unfolded', norm(herwig_unfolded[sl],  r_edges), C_HERWIG_UNF,  ':',  1.0),
        ('Pythia Reco',     norm(pythia_reco[sl],      r_edges), C_PYTHIA_RECO, '--', 1.0),
        ('Pythia Unfolded', norm(pythia_unfolded[sl],  r_edges), C_PYTHIA_UNF,  ':',  1.0),
    ]

    ref = curves[1][1]

    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(10, 13),
        gridspec_kw={'height_ratios': [10, 3]},
        sharex=True
    )

    for label, vals, color, ls, lw in curves:
        hep.histplot(vals, r_edges, ax=ax, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.set_xscale('log')
    if i > 0:
        ax.set_yscale('log')
    ax.set_ylabel('A.U.', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(f'Jet $p_T \\in [{jlo}, {jhi}]$ GeV', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    hep.cms.label(ax=ax, data=True, text='Private', com=13)

    for label, vals, color, ls, lw in curves:
        if vals is ref:
            continue
        ratio = np.where(ref != 0, vals / ref, np.nan)
        hep.histplot(ratio, r_edges, ax=ax_ratio, label=label, color=color, linestyle=ls, linewidth=lw)

    ax_ratio.axhline(1.0, color=C_PYTHIA_GEN, linestyle='-', linewidth=0.8)
    ax_ratio.set_xscale('log')
    ax_ratio.set_xlabel(r'$\Delta R$', fontsize=11)
    ax_ratio.set_ylabel('Ratio to Data Reco', fontsize=11)
    ax_ratio.tick_params(axis='both', labelsize=10)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(args.output, f'unfolded_Jpt{jlo}-{jhi}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print('Saved', fname)
