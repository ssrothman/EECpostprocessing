from email.utils import collapse_rfc2231_value

from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack, load_prebinned_dataset, build_prebinned_dataset_stack, load_prebinned_root_histogram
import simonplot as splt
from simonplot.binning.Binning import PrebinnedBinning
from simonplot.cut.CutBase import PrebinnedOperationBase
import json
import os
import numpy as np

from simonplot.plottables.PrebinnedDatasets import ValCovPairDataset
from simonplot.typing.Protocols import PrebinnedVariableProtocol, VariableProtocol
from simonplot.util.evaluate import evaluate_on_dataset

def build_prebinned_comparison_dataset(dscfg1 : dict, dscfg2 : dict, func : str, thevariable : PrebinnedVariableProtocol, key : str | None= None, color : str | None= None, label : str | None = None):
    dset1= build_dataset_from_dscfg(
        dscfg1,
        True,
        True,
        splt.cut.NoCut()
    )

    dset2 = build_dataset_from_dscfg(
        dscfg2,
        True,
        True,
        splt.cut.NoCut()
    )

    if dset1.binning != dset2.binning:
        raise ValueError("Cannot build a comparison dataset from two datasets with different binnings!")
    if dset1.isMC != dset2.isMC:
        raise ValueError("Cannot build a comparison dataset from two datasets where one is MC and the other is not!")
    
    val1, cov1 = evaluate_on_dataset(dset1, thevariable, splt.cut.NoopOperation())
    val2, cov2 = evaluate_on_dataset(dset2, thevariable, splt.cut.NoopOperation())

    print(func)

    # assume the two datasets are uncorrelated, so we add the covariance matrices
    if func == 'sum':
        newval = val1 + val2
        newcov = cov1 + cov2
    elif func == 'difference':
        newval = val1 - val2
        newcov = cov1 + cov2
    elif func == 'product':
        newval = val1 * val2
        newcov = np.outer(val2, val2) * cov1 + np.outer(val1, val1) * cov2
    elif func == 'ratio':
        newval = val1 / val2
        newcov = np.outer(1/val2, 1/val2) * cov1 + np.outer(val1/(val2*val2), val1/(val2*val2)) * cov2
    else:
        raise ValueError(f"Unsupported func {func} for building comparison dataset! Supported funcs are: 'sum', 'difference', 'product', and 'ratio'.")
    

    if key is None:
        thekey = f"{dset1.key}_{func}_{dset2.key}"
    else:
        thekey = key

    if label is None:
        thelabel = f"{dset1.label} {func} {dset2.label}"
    else:       
        thelabel = label

    if color is None:
        thecolor = dset1.color
    else:        
        thecolor = color
    
    result = ValCovPairDataset(
        key = thekey,
        label = thelabel,
        color = thecolor,
        data = (newval, newcov),
        binning = dset1.binning,
        isMC = dset1.isMC
    )
    return result

def build_dataset_from_dscfg(dscfg, all_same_objsyst, all_same_extracut, dsetcut):
    if dscfg['dsetkind'] == 'prebinned_comparison':
        return build_prebinned_comparison_dataset(
            dscfg['dataset1'],
            dscfg['dataset2'],
            dscfg['comparison'],
            thevariable = parse_var(dscfg['thevariable']), # type: ignore
            key=dscfg.get('key', None),
            color=dscfg.get('color_override', None),
            label=dscfg.get('label_override', None)
        )
    elif dscfg['dsetkind'] == 'prebinned_stack':
        factory = build_prebinned_dataset_stack
    elif dscfg['dsetkind'] == 'prebinned':
        factory = load_prebinned_dataset
    elif dscfg['dsetkind'] == 'prebinned_root_histogram':
        factory = load_prebinned_root_histogram
    elif dscfg['dsetkind'] == 'pq_stack':
        factory = build_pq_dataset_stack
    elif dscfg['dsetkind'] == 'pq':
        factory = build_pq_dataset
    else:
        raise ValueError(f"Unknown dsetkind {dscfg['dsetkind']} in dataset config!")
    
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

    if not 'prebinned' in dscfg['dsetkind']:
        extraargs['no_count'] = dscfg.get('no_count', False) or dscfg.get('nocount', False) # support both spellings of this option for now

    if 'prebinned' in dscfg['dsetkind']:
        extraargs['wtsyst'] = dscfg.get('wtsyst', 'nominal')

    if 'nocov' in dscfg:
        extraargs['nocov'] = dscfg['nocov']

    if 'path' in dscfg:
        extraargs['path'] = dscfg['path']

    if 'statN' in dscfg and 'statK' in dscfg:
        extraargs['statN'] = dscfg['statN']
        extraargs['statK'] = dscfg['statK']

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

        if dscfg['dsetkind'] == 'unbinned_comparison':
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

    if 'override_cuttext' in cfg:
        if isinstance(cut, list):
            for onecut in cut:
                onecut.override_label(cfg['override_cuttext']) # pyright: ignore[reportAttributeAccessIssue]
        else:
            cut.override_label(cfg['override_cuttext']) # pyright: ignore[reportAttributeAccessIssue]

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
    elif cfg['driver'] == 'draw_radial_histogram':
        if len(variables) != 1:
            raise RuntimeError("draw_radial_histogram only supports exactly one variable")
        if not isinstance(binning, PrebinnedBinning):
            raise RuntimeError("draw_radial_histogram only supports prebinned binning")

        if isinstance(cut, list):
            if len(cut) != 1:
                raise RuntimeError("draw_radial_histogram only supports exactly one cut")
            radial_cut = cut[0]
        else:
            radial_cut = cut

        override_filenames = cfg.get('override_filenames', [])

        override_cbarlabels = cfg.get('override_cbarlabel', [])

        extratexts = cfg.get('extratext', [])

        for i, dataset in enumerate(datasets):
            if not isinstance(radial_cut, PrebinnedOperationBase):
                raise RuntimeError("draw_radial_histogram only supports prebinned cuts")

            radial_override = override_filenames[i] if i < len(override_filenames) else None
            cbarlabel = override_cbarlabels[i] if i < len(override_cbarlabels) else None
            extratext = extratexts[i] if i < len(extratexts) else None

            print(f"Plotting radial variable {variables[0].key} for dataset {dataset.key}")
            splt.draw_radial_histogram(
                variables[0],
                radial_cut,
                dataset,
                binning,
                extratext=extratext,
                logc=cfg.get('logc', None),
                sym=cfg.get('sym', None),
                override_cbarlabel=cbarlabel,
                output_folder=cfg['plotspath'],
                output_prefix=cfg['plotsprefix'],
                override_filename=radial_override,
            )
    else:
        raise NotImplementedError(f"Plotting driver {cfg['driver']} not implemented yet in this driver script!")