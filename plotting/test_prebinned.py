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

vars = [var_basic, var_w_jac, var_normalized, var_normalized_w_jac]

cut = splt.cut.SliceOperation(
    edges = {
        'R' : [0.4, 0.5],
    },
    clipemptyflow = ['r', 'c']
)

weight = splt.variable.ConstantVariable(1.0)
binning = splt.binning.PrebinnedBinning()

#for var in vars:
#    print(f"Plotting variable {var.key}")
#    splt.plot_histogram(
#        var,
#        cut,
#        weight,
#        herwig_glu,
#        binning,
#        output_folder='testplots/prebinned',
#        output_prefix='evt',
#    )

cut = splt.cut.SliceOperation(
    edges = {
        'Jpt' : [300, 500],
        'R' : [0.4, 0.5],
    },
    clipemptyflow = ['r', 'c']
)
for var in vars[:2]:
    splt.draw_radial_histogram(
        var,
        cut,
        herwig_glu,
        binning,
        output_folder='testplots/prebinned',
        output_prefix='evt_radial',
    )