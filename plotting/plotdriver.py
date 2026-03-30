from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack, load_prebinned_dataset, build_prebinned_dataset_stack
import simonplot as splt
import json
import os


def build_dataset_from_dscfg(dscfg, all_same_objsyst, all_same_extracut, dsetcut):
    if dscfg['isstack']:
        if dscfg.get('isprebinned', False):
            factory = build_prebinned_dataset_stack
        else:
            factory = build_pq_dataset_stack
    else:
        if dscfg.get('isprebinned', False):
            factory = load_prebinned_dataset
        else:
            factory = build_pq_dataset

    extrakey = ''
    if not all_same_objsyst:
        extrakey += dscfg['objsyst'] + '-'
    if not all_same_extracut and not isinstance(dsetcut, splt.cut.NoCut):
        extrakey += dsetcut.key + '-'
    if extrakey.endswith('-'):
        extrakey = extrakey[:-1]

    extraargs = {}
    if 'never_resolve' in dscfg:
        extraargs['showStack'] = not dscfg['never_resolve']

    if not dscfg.get('isprebinned', False):
        extraargs['no_count'] = dscfg.get('no_count', False)

    if dscfg.get('isprebinned', False):
        extraargs['wtsyst'] = dscfg.get('wtsyst', 'nominal')

    if 'nocov' in dscfg:
        extraargs['nocov'] = dscfg['nocov']

    return factory(
        configsuite=dscfg['configsuite'],
        runtag=dscfg['runtag'],
        dataset=dscfg['name'],
        objsyst=dscfg['objsyst'],
        table=dscfg['table'],
        location=dscfg.get('location', 'xrootd-submit'),
        label_override=dscfg.get('label_override', None),
        color_override=dscfg.get('color_override', None),
        extra_key=extrakey if extrakey else None,
        **extraargs
    )

def parse_var(varname):
    if '::' in varname:
        varname = varname.replace('cut::', 'splt.cut.')
        varname = varname.replace('var::', 'splt.variable.')
        varname = varname.replace('func::', 'splt.plottables.Functions.')
        if '::' in varname:
            raise ValueError(f"Variable {varname} has an unknown prefix!")
        return eval(varname)
    else:
        return splt.variable.BasicVariable(varname)

def run_plots(cfg):
    datasets = []
    dsetcuts = []
    nExtraCuts = 0

    #check if all objsysts are the same
    if 'comparison' in cfg['datasets'][0]:
        first_objsyst = cfg['datasets'][0]['dataset1']['objsyst']
    else:
        first_objsyst = cfg['datasets'][0]['objsyst']
        
    all_same_objsyst = True
    for dsetcfg in cfg['datasets'][1:]:
        if 'comparison' in dsetcfg:
            if dsetcfg['dataset1']['objsyst'] != first_objsyst or dsetcfg['dataset2']['objsyst'] != first_objsyst:
                all_same_objsyst = False
                break
        else:
            if dsetcfg['objsyst'] != first_objsyst:
                all_same_objsyst = False
                break

    #check if extracuts are all the same
    first_extracut = cfg['datasets'][0].get('extra_cuts', [])
    all_same_extracut = True
    for dsetcfg in cfg['datasets'][1:]:
        if dsetcfg.get('extra_cuts', []) != first_extracut:
            all_same_extracut = False
            break

    for dscfg in cfg['datasets']:
        if 'extra_cuts' in dscfg and len(dscfg['extra_cuts']) > 0:
            thecuts = []
            for cut in dscfg['extra_cuts']:
                thecuts.append(parse_var(cut))
            dsetcuts.append(splt.cut.AndCuts(thecuts))
            nExtraCuts += 1
        else:
            dsetcuts.append(splt.cut.NoCut())

        if 'comparison' in dscfg:
            dset1 = build_dataset_from_dscfg(
                dscfg['dataset1'],
                all_same_objsyst,
                all_same_extracut,
                dsetcuts[-1],
            )
            dset2 = build_dataset_from_dscfg(
                dscfg['dataset2'],
                all_same_objsyst,
                all_same_extracut,
                dsetcuts[-1],
            )
            datasets.append(splt.plottables.DatasetComparison(
                key=dscfg['key'],
                color=dscfg.get('color_override', None),
                label=dscfg.get('label_override', dscfg['key']),
                ylabel = dscfg['ylabel'],
                dataset1=dset1,
                dataset2=dset2,
                kind=dscfg['comparison']
            ))
        else:
            datasets.append(build_dataset_from_dscfg(
                dscfg,
                all_same_objsyst,
                all_same_extracut,
                dsetcuts[-1],
            ))

    
    variables = []
    for varname in cfg['variables']:
        variables.append(parse_var(varname))

    weights = []
    for wname in cfg['weights']:
        weights.append(parse_var(wname))
    if len(cfg['cut']) == 0:
        cut = splt.cut.NoCut()
    else:
        thecuts = []
        for cut in cfg['cut']:
            thecuts.append(parse_var(cut))
        cut = splt.cut.AndCuts(thecuts)

    if nExtraCuts > 0:
        cut = [
            splt.cut.AndCuts([cut, dsetcut]) for dsetcut in dsetcuts
        ]

    if cfg['binning'] == 'auto':
        binning = splt.binning.AutoBinning()
    elif cfg['binning'].startswith('autoint:'):
        labelkey = cfg['binning'].split(':')[1]
        with open(os.path.join(os.path.dirname(__file__), 'autoint_lookups.json'), 'r') as f:
            label_lookup = json.load(f)
    
        binning = splt.binning.AutoIntCategoryBinning(
            label_lookup=label_lookup.get(labelkey, {})
        )
    elif cfg['binning'].startswith('regular'):
        #expect something of the form "regular:(start, stop, nbins)"
        _, params = cfg['binning'].split(':')
        start, stop, nbins = eval(params)
        binning = splt.binning.BasicBinning(nbins=nbins, low=start, high=stop)
    elif cfg['binning'] == 'prebinned':
        binning = splt.binning.PrebinnedBinning()
    else:
        raise NotImplementedError("Binning type {} not implemented in this driver script!".format(cfg['binning']))
    
    if 'force_range' in cfg:
        if hasattr(binning, 'force_range'):
            binning.force_range(*cfg['force_range']) # pyright: ignore[reportAttributeAccessIssue]
        else:
            raise ValueError(f"binning {binning} does not support force_range")

    if cfg['plotsprefix'] == '':
        cfg['plotsprefix'] = None

    extra_stuff = []
    if 'extras' in cfg:
        for extra in cfg['extras']:
            extra_stuff.append(parse_var(extra))

    if cfg['driver'] == 'plot_histogram':
        for i, var in enumerate(variables):
            print(f"Plotting variable {var.key}")
            splt.plot_histogram(
                var,
                cut,
                weights,
                datasets,
                binning,
                output_folder=cfg['plotspath'],
                output_prefix=cfg['plotsprefix'],
                no_ratiopad=cfg.get('nopad', False),
                no_lumi_normalization=cfg.get('no_lumi_normalization', False),
                logy=cfg.get('logy', None),
                density=cfg.get('density', False),
                override_filename=cfg.get('override_filenames', [None]*len(variables))[i],
                extra_stuff=extra_stuff
            )
    else:
        raise NotImplementedError(f"Plotting driver {cfg['driver']} not implemented yet in this driver script!")