import simonplot as splt
from plotting.load_datasets import load_prebinned_dataset
from simonplot.cut.common_cuts import common_cuts

herwig_glu = load_prebinned_dataset(
    'GenonlyConfig',
    'Feb_15_2026',
    'herwig_glu',
    'nominal',
    'nominal',
    'res4tee_totalReco',
    location='scratch-submit'
)

herwig_glu_nospin = load_prebinned_dataset(
    'GenonlyConfig',
    'Feb_15_2026',
    'herwig_glu_nospin_8X',
    'nominal',
    'nominal',
    'res4tee_totalReco',
    location='scratch-submit'
)

pythia_glu = load_prebinned_dataset(
    'GenonlyConfig',
    'Feb_15_2026',
    'pythia_glu',
    'nominal',
    'nominal',
    'res4tee_totalReco',
    location='scratch-submit'
)

pythia_glu_nospin = load_prebinned_dataset(
    'GenonlyConfig',
    'Feb_15_2026',
    'pythia_glu_nospin',
    'nominal',
    'nominal',
    'res4tee_totalReco',
    location='scratch-submit'
)



cut = splt.cut.SliceOperation(
    edges = {
        'Jpt' : [1200, 2000],
        'R' : [0.4, 0.5],
        #'r' : [0.05, 0.10]
    },
    clipemptyflow=['c']
)
var = splt.variable.BasicPrebinnedVariable()
wt = splt.variable.ConstantVariable(1.0)
binning = splt.binning.PrebinnedBinning()

splt.plot_histogram(
    var,
    cut,
    wt,
    [pythia_glu, pythia_glu_nospin],
    binning,
    output_folder='testplots/genonly5/',
    output_prefix='res4tee',
)