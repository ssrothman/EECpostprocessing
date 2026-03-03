import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack

print("building herwig_glu dataset")
herwig_glu = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu',
    'nominal',
    'GenSplittings',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_nospin dataset")
herwig_glu_nospin = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_nospin',
    'nominal',
    'GenSplittings',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_gg dataset")
herwig_glu_gg = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_gg',
    'nominal',
    'GenSplittings',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_nospin_gg dataset")
herwig_glu_nospin_gg = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_nospin_gg',
    'nominal',
    'GenSplittings',
    location='scratch-submit',
    no_count=True
)

#cut = splt.cut.NoCut()

weight = splt.variable.BasicVariable('wt_nominal')
binning = splt.binning.AutoBinning()

dpsi = splt.variable.BasicVariable('deltaPsi')
cut = splt.cut.AndCuts([
    splt.cut.GreaterThanCut(dpsi, -10),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('pt1'), 20),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('z23'), 0.1),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('z45'), 0.1)
])

print("plotting")

binning2 = splt.binning.AutoIntCategoryBinning(
    label_lookup = {
        '-1' : 'Unknown',
        '0' : 'g -> g(g->gg)',
        '4' : 'g -> g(g->qq)',
        '9' : 'g -> q(q->qg)',
        '13' : 'g -> q(q->gq)',
        '3' : 'q -> q(g->gg)',
        '7' : 'q -> q(g->qq)',
        '10' : 'q -> g(q->qg)',
        '14' : 'q -> g(q->gq)',
    }
)
var2 = splt.variable.BasicVariable('splitType')
splt.plot_histogram(
    var2,
    cut,
    weight,
    [
        herwig_glu,
        herwig_glu_nospin,
        herwig_glu_gg,
        herwig_glu_nospin_gg
    ],
    binning2,
    output_folder='testplots/genonly2',
    output_prefix='jet',
    #logy= not ('psi' in varname.lower() or 'phi' in varname.lower()),
    #logx = 'pt' in varname.lower() or 'kt' in varname.lower() or '_z' in varname.lower()
)

for varname in herwig_glu.schema.names:
    var = splt.variable.BasicVariable(varname)
    splt.plot_histogram(
        var,
        cut,
        weight,
        [
            herwig_glu,
            herwig_glu_nospin,
            herwig_glu_gg,
            herwig_glu_nospin_gg
        ],
        binning,
        output_folder='testplots/genonly2',
        output_prefix='jet',
        logy= not ('psi' in varname.lower() or 'phi' in varname.lower()),
        logx = 'pt' in varname.lower() or 'kt' in varname.lower() or '_z' in varname.lower()
    )