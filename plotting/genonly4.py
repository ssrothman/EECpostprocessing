import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
from simonplot.cut.common_cuts import common_cuts

dsets_l = [
    ['herwig_glu', 'herwig_glu_nospin_5X', 'herwig_glu_nospin_6X', 'herwig_glu_nospin_8X'],
    ['pythia_glu', 'pythia_glu_nospin']
]

for dsets in dsets_l:
    for table in ['GenSplittingsHardSide', 'GenSplittingsSoftSide', 'DeltaPsiHardSide', 'DeltaPsiSoftSide']:
        pqdsets = [
            build_pq_dataset(
                'GenonlyConfig',
                'Feb_15_2026',
                dset,
                'nominal',
                table,
                location='scratch-submit',
                no_count=True
            )
            for dset in dsets
        ]
        
        weight = splt.variable.BasicVariable('wt_nominal')
        binning = splt.binning.AutoBinning()

        flavcut = splt.cut.AllEqualCut([
            splt.variable.BasicVariable('pdgId1'),
            splt.variable.BasicVariable('pdgId2'),
            splt.variable.BasicVariable('pdgId3'),
            splt.variable.BasicVariable('pdgId4'),
            splt.variable.BasicVariable('pdgId5'),
            splt.variable.BasicVariable('pdgId6')
        ], 21)
        flavcut.override_label('g->g(g->gg)')

        if 'GenSplittings' in table:
            cut = splt.cut.AndCuts([
                flavcut,
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('pt1'), 500),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('pt4'), 100),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('z23'), 0.1),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('z56'), 0.2),
                #splt.cut.LessThanCut(splt.variable.BasicVariable('deltaR23'), 0.1)
            ])
        else:
            cut = splt.cut.AndCuts([
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('pt1'), 500),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('pt4'), 100),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('z23'), 0.1),
                splt.cut.GreaterThanCut(splt.variable.BasicVariable('z56'), 0.2),
                #splt.cut.LessThanCut(splt.variable.BasicVariable('deltaR23'), 0.1)
            ])

        for varname in pqdsets[0].schema.names:
        #for varname in ['deltaPsi_type1', 'deltaPsi_type3']:
            print("Plotting variable %s"%varname)
            var = splt.variable.BasicVariable(varname)
            splt.plot_histogram(
                var,
                cut,
                weight,
                pqdsets,
                binning,
                output_folder='testplots/genonly4/%s'%table,
                output_prefix='jet',
                logy= not ('psi' in varname.lower() or 'phi' in varname.lower()),
                logx = 'pt' in varname.lower() or 'kt' in varname.lower() or 'z' in varname.lower() or 'c2' in varname.lower()
            )