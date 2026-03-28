from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack, load_prebinned_dataset
import simonplot as splt
import json
import os

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
    first_objsyst = cfg['datasets'][0]['objsyst']
    all_same_objsyst = True
    for dsetcfg in cfg['datasets'][1:]:
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
        
        if dscfg.get('prebinned', False):
            datasets.append(
                load_prebinned_dataset(
                    dscfg['configsuite'],
                    dscfg['runtag'],
                    dscfg['name'],
                    dscfg['objsyst'],
                    dscfg.get('wtsyst', 'nominal'),
                    dscfg['table'],
                    location=dscfg.get('location', 'xrootd-submit')
                )
            )
        else:
            if dscfg['isstack']:
                factory = build_pq_dataset_stack
            else:
                factory = build_pq_dataset

            extrakey = ''
            if not all_same_objsyst:
                extrakey += dscfg['objsyst'] + '-'
            if not all_same_extracut and not isinstance(dsetcuts[-1], splt.cut.NoCut):
                extrakey += dsetcuts[-1].key + '-'
            if extrakey.endswith('-'):
                extrakey = extrakey[:-1]

            datasets.append(
                factory(
                    dscfg['configsuite'],
                    dscfg['runtag'],
                    dscfg['name'],
                    dscfg['objsyst'],
                    dscfg['table'],
                    dscfg.get('location', 'xrootd-submit'),
                    no_count = dscfg.get('no_count', False),
                    label_override=dscfg.get('label_override', None),
                    color_override=dscfg.get('color_override', None),
                    extra_key = extrakey if extrakey else None
                )
            )

    
    variables = []
    for varname in cfg['variables']:
        variables.append(parse_var(varname))

    weights = []
    for wname in cfg['weights']:
        weights.append(parse_var(wname))
    def parse_cut(c):
        if isinstance(c, dict) and c.get('type') == 'slice':
            return splt.cut.SliceOperation(
                edges=c['edges'],
                clipemptyflow=c.get('clipemptyflow', [])
            )
        return parse_var(c)

    if len(cfg['cut']) == 0:
        cut = splt.cut.NoCut()
    else:
        thecuts = []
        for c in cfg['cut']:
            thecuts.append(parse_cut(c))
        cut = splt.cut.AndCuts(thecuts)

    if nExtraCuts > 0:
        cut = [
            splt.cut.AndCuts([cut, dsetcut]) for dsetcut in dsetcuts
        ]

    if cfg['binning'] == 'auto':
        binning = splt.binning.AutoBinning()
    elif cfg['binning'] == 'prebinned':
        binning = splt.binning.PrebinnedBinning()
    elif cfg['binning'].startswith('explicit:'):
        edges = json.loads(cfg['binning'].split(':', 1)[1])
        binning = splt.binning.ExplicitBinning(edges)
    elif cfg['binning'].startswith('autoint:'):
        labelkey = cfg['binning'].split(':')[1]
        with open(os.path.join(os.path.dirname(__file__), 'autoint_lookups.json'), 'r') as f:
            label_lookup = json.load(f)
        binning = splt.binning.AutoIntCategoryBinning(
            label_lookup=label_lookup.get(labelkey, {})
        )
    else:
        raise NotImplementedError(f"Binning '{cfg['binning']}' not implemented in this driver script")
    
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
                logy=cfg.get('logy', None),
                logx=cfg.get('logx', False),
                density=cfg.get('density', False),
                no_lumi_normalization=cfg.get('no_lumi_normalization', False),
                override_filename=cfg.get('override_filenames', [None]*len(variables))[i],
                extra_stuff=extra_stuff
            )
    else:
        raise NotImplementedError(f"Plotting driver {cfg['driver']} not implemented yet in this driver script!")