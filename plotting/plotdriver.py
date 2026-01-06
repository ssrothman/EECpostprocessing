from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import simonplot as splt
import json
import os

def parse_var(varname):
    if '::' in varname:
        varname = varname.replace('cut::', 'splt.cut.')
        varname = varname.replace('var::', 'splt.variable.')
        if '::' in varname:
            raise ValueError(f"Variable {varname} has an unknown prefix!")
        return eval(varname)
    else:
        return splt.variable.BasicVariable(varname)

def run_plots(cfg):
    datasets = []
    for dscfg in cfg['datasets']:
        if dscfg['isstack']:
            factory = build_pq_dataset_stack
        else:
            factory = build_pq_dataset
        
        datasets.append(
            factory(
                dscfg['configsuite'],
                dscfg['runtag'],
                dscfg['name'],
                dscfg['objsyst'],
                dscfg['table']
            )
        )
    
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

    if cfg['binning'] == 'auto':
        binning = splt.binning.AutoBinning()
    elif cfg['binning'].startswith('autoint:'):
        labelkey = cfg['binning'].split(':')[1]
        with open(os.path.join(os.path.dirname(__file__), 'autoint_lookups.json'), 'r') as f:
            label_lookup = json.load(f)
    
        binning = splt.binning.AutoIntCategoryBinning(
            label_lookup=label_lookup.get(labelkey, {})
        )
    else:
        raise NotImplementedError("Only 'auto' binning is implemented so far in this driver script")
    
    if cfg['plotsprefix'] == '':
        cfg['plotsprefix'] = None

    if cfg['driver'] == 'plot_histogram':
        for var in variables:
            print(f"Plotting variable {var.key}")
            splt.plot_histogram(
                var,
                cut,
                weights,
                datasets,
                binning,
                output_folder=cfg['plotspath'],
                output_prefix=cfg['plotsprefix'],
            )
    else:
        raise NotImplementedError(f"Plotting driver {cfg['driver']} not implemented yet in this driver script!")