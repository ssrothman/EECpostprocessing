import simonplot as splt
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
from simonplot.cut.common_cuts import common_cuts

for MC in ['pythia', 'herwig']:
    for table in ['GenSplittingsHardSide', 'GenSplittingsSoftSide', 'DeltaPsiHardSide', 'DeltaPsiSoftSide']:
        glu = build_pq_dataset(
            'GenonlyConfig',
            'Feb_11_2026',
            f'{MC}_glu_TeV',
            'nominal',
            table,
            location='scratch-submit',
            no_count=True
        )

        glu_nospin = build_pq_dataset(
            'GenonlyConfig',
            'Feb_11_2026',
            f'{MC}_glu_TeV_nospin',
            'nominal',
            table,
            location='scratch-submit',
            no_count=True
        )

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

        for varname in glu.schema.names:
            print("Plotting variable %s"%varname)
            var = splt.variable.BasicVariable(varname)
            splt.plot_histogram(
                var,
                cut,
                weight,
                [
                    glu,
                    glu_nospin
                ],
                binning,
                output_folder='testplots/genonly3/%s'%table,
                output_prefix='jet',
                logy= not ('psi' in varname.lower() or 'phi' in varname.lower()),
                logx = 'pt' in varname.lower() or 'kt' in varname.lower() or 'z' in varname.lower()
            )