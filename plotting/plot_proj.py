import simonplot as splt
import numpy as np
from plotting.load_datasets import build_pq_dataset, load_prebinned_dataset

LOCATION    = 'dylan-lxplus-eos'
CONFIG      = 'EvtMCprojConfig'
RUNTAG      = 'new_v3'
OBJSYST     = 'NOM'
WTSYST      = 'nominal'
OUTPUT      = 'plots/proj'

herwig_reco = load_prebinned_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Herwig', OBJSYST, WTSYST, 'proj_Reco',
    location=LOCATION
)
pythia_reco = load_prebinned_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Pythia', OBJSYST, WTSYST, 'proj_Reco',
    location=LOCATION
)
herwig_gen = load_prebinned_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Herwig', OBJSYST, WTSYST, 'proj_Gen',
    location=LOCATION
)
pythia_gen = load_prebinned_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Pythia', OBJSYST, WTSYST, 'proj_Gen',
    location=LOCATION
)

datasets_reco = [herwig_reco, pythia_reco]
datasets_gen  = [herwig_gen,  pythia_gen]

var     = splt.variable.BasicPrebinnedVariable()
weight  = splt.variable.ConstantVariable(1.0)
binning = splt.binning.PrebinnedBinning()

jpt_bins = [40, 100, 200, 340, 520, 740, 1000]
for lo, hi in zip(jpt_bins[:-1], jpt_bins[1:]):
    cut = splt.cut.SliceOperation(
        edges={'Jpt': [lo, hi]},
        clipemptyflow=[]
    )
    for level, dsets in [('reco', datasets_reco), ('gen', datasets_gen)]:
        splt.plot_histogram(
            var, cut, weight, dsets, binning,
            output_folder=OUTPUT,
            output_prefix='EEC_%s_Jpt%d-%d' % (level, lo, hi),
            no_lumi_normalization=True,
            logx=True,
        )

# Jpt distribution from direct parquet
herwig_reco_pq = build_pq_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Herwig', OBJSYST, 'proj_Reco',
    location=LOCATION, no_count=True
)
pythia_reco_pq = build_pq_dataset(
    CONFIG, RUNTAG, 'DYJetsToLL_Pythia', OBJSYST, 'proj_Reco',
    location=LOCATION, no_count=True
)

jpt_var    = splt.variable.BasicVariable('Jpt')
jpt_weight = splt.variable.BasicVariable('wt_nominal')
jpt_cut    = splt.cut.NoCut()
jpt_binning = splt.binning.ExplicitBinning(list(np.linspace(4, 1000, 101)))

splt.plot_histogram(
    jpt_var, jpt_cut, jpt_weight,
    [herwig_reco_pq, pythia_reco_pq],
    jpt_binning,
    output_folder=OUTPUT,
    output_prefix='Jpt_reco',
    no_lumi_normalization=True,
    logx=True,
)
