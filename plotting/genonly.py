import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack

print("building herwig_glu dataset")
herwig_glu = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu',
    'nominal',
    'jets',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_nospin dataset")
herwig_glu_nospin = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_nospin',
    'nominal',
    'jets',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_gg dataset")
herwig_glu_gg = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_gg',
    'nominal',
    'jets',
    location='scratch-submit',
    no_count=True
)

print("building herwig_glu_nospin_gg dataset")
herwig_glu_nospin_gg = build_pq_dataset(
    'GenonlyConfig',
    'Feb_09_2026',
    'herwig_glu_nospin_gg',
    'nominal',
    'jets',
    location='scratch-submit',
    no_count=True
)

#cut = splt.cut.NoCut()

weight = splt.variable.BasicVariable('wt_nominal')
binning = splt.binning.AutoBinning()

dpsi = splt.variable.BasicVariable('splitting_deltaPsi')
cut = splt.cut.AndCuts([
    splt.cut.GreaterThanCut(dpsi, 0),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('splitting_pt1'), 20),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('splitting_z23'), 0.1),
    splt.cut.GreaterThanCut(splt.variable.BasicVariable('splitting_z45'), 0.1)
])

print("plotting")

for varname in herwig_glu.schema.names:
    if varname.startswith('splitting'):
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
            output_folder='testplots/genonly',
            output_prefix='jet',
            logy= not ('psi' in varname.lower() or 'phi' in varname.lower()),
            logx = 'pt' in varname.lower() or 'kt' in varname.lower() or '_z' in varname.lower()
        )