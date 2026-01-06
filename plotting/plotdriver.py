from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import simonplot as splt

def parse_variable(varname):
    if 'splt::' in varname:
        varname = varname.replace('splt::', 'splt.variable.')
        return eval(varname)        
    else:
        return splt.variable.BasicVariable(varname)
    
def parse_cut(varname):
    if 'splt::' in varname:
        varname = varname.replace('splt::', 'splt.cut.')
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
        variables.append(parse_variable(varname))

    weights = []
    for wname in cfg['weights']:
        weights.append(parse_variable(wname))

    if len(cfg['cut']) == 0:
        cut = splt.cut.NoCut()
    else:
        thecuts = []
        for cut in cfg['cut']:
            thecuts.append(parse_cut(cut))
        cut = splt.cut.AndCuts(thecuts)

    if cfg['binning'] == 'auto':
        binning = splt.binning.AutoBinning()
    else:
        raise NotImplementedError("Only 'auto' binning is implemented so far in this driver script")
    
    if cfg['plotsprefix'] == '':
        cfg['plotsprefix'] = None

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