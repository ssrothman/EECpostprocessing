import simonplot as splt
from plotting.load_datasets import load_prebinned_dataset
import numpy as np

herwig_glu = load_prebinned_dataset(
    'GenonlyConfig',
    'May_14_2025',
    'herwig_glu',
    'nominal',
    'nominal',
    'res4tee_totalReco'
)
herwig_glu_nospin = load_prebinned_dataset(
    'GenonlyConfig',
    'May_14_2025',
    'herwig_glu_nospin',
    'nominal',
    'nominal',
    'res4tee_totalReco'
)
herwig_glu_TeV = load_prebinned_dataset(
    'GenonlyConfig',
    'May_14_2025',
    'herwig_glu_TeV',
    'nominal',
    'nominal',
    'res4tee_totalReco'
)
herwig_glu_TeV_nospin = load_prebinned_dataset(
    'GenonlyConfig',
    'May_14_2025',
    'herwig_glu_TeV_nospin',
    'nominal',
    'nominal',
    'res4tee_totalReco'
)

var_basic = splt.variable.BasicPrebinnedVariable()
var_w_jac = splt.variable.WithJacobian(
    var_basic,
    wrt = ['r', 'c'],
    radial_coords = ['r'],
)
var_normalized = splt.variable.NormalizePerBlock(
    var_basic,
    axes = ['Jpt']
)
var_normalized_w_jac = splt.variable.WithJacobian(
    var_normalized,
    wrt = ['r', 'c'],
    radial_coords = ['r'],
)
var_normr = splt.variable.DivideOutProfile(
    var_normalized_w_jac,
    ['Jpt', 'r']
)

vars = [
    var_basic,
    var_w_jac,
    var_normalized, 
    var_normalized_w_jac, 
    var_normr
]

cut_basic = splt.cut.SliceOperation(
    edges = {
        'R' : [0.4, 0.5],
        'r' : [0.0, 0.2]
    },
    clipemptyflow = []
)

cut = splt.cut.SliceOperation(
    edges = {
        'R' : [0.4, 0.5],
        'r' : [0.0, 0.2]
    },
    clipemptyflow = [
        'r', 'c'
    ]
)

weight = splt.variable.ConstantVariable(1.0)
binning = splt.binning.PrebinnedBinning()

var_corr = splt.variable.CorrelationFromCovariance(var_basic)

splt.draw_matrix(
    var_corr,
    cut_basic,
    herwig_glu,
    binning,
    output_folder='testplots/prebinned',
    output_prefix='evt_matrix_basic',
)

for var in vars:
    print(f"Plotting variable {var.key}")
    splt.plot_histogram(
        var,
        cut,
        weight,
        [
            herwig_glu,
            herwig_glu_nospin,
            herwig_glu_TeV,
            herwig_glu_TeV_nospin
        ],
        binning,
        output_folder='testplots/prebinned',
        output_prefix='evt',
        logy=None,
        no_lumi_normalization=True,
    )

cut = splt.cut.SliceOperation(
    edges = {
        'Jpt' : [300, 500],
        'R' : [0.4, 0.5],
    },
    clipemptyflow = ['r', 'c']
)

var_basic = splt.variable.BasicPrebinnedVariable()
var_w_jac = splt.variable.WithJacobian(
    var_basic,
    wrt = ['r', 'c'],
    radial_coords = ['r'],
)
var_normr = splt.variable.DivideOutProfile(
    var_w_jac,
    ['r']
)
vars = [var_basic, var_w_jac, var_normr]
for var in vars:
    splt.draw_radial_histogram(
        var,
        cut,
        herwig_glu,
        binning,
        output_folder='testplots/prebinned',
        output_prefix='evt_radial',
    )