from reweighting.reweighting_backend import compare_smoothings, dump_smoothings
from reweighting.smoothing import SplineSmoothing

from .spec import dset_spec, reweighting_spec
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
import simonplot as splt
import numpy as np
import pyarrow.compute as pc
from simonpy.parse_splt import parse_splt
from typing import Sequence
import os.path

def build_dataset(spec : dset_spec):
    if spec['isstack']:
        factory = build_pq_dataset_stack
    else:
        factory = build_pq_dataset

    return factory(
        configsuite = spec['config'],
        runtag = spec['runtag'],
        dataset = spec['dataset'],
        objsyst = spec['objsyst'],
        table = spec['table'],
        location = spec['location']
    ), splt.variable.BasicVariable(spec['wtsyst'])

def parse_andcuts(cuts : Sequence[str]):
    if len(cuts) == 0:
        return splt.cut.NoCut()
    else:
        return splt.cut.AndCuts([
            parse_splt(cut) for cut in cuts
        ])

def run_reweighting(spec :reweighting_spec):
    dset_num, wt_num = build_dataset(spec['num'])
    dset_denom, wt_denom = build_dataset(spec['denom'])

    if dset_num.isMC and dset_denom.isMC:
        dset_num.compute_weight(1.0)
        dset_denom.compute_weight(1.0)
        isMC = True
        lumi = None
    else:
        isMC = False

        if dset_num.isMC:
            dset_num.compute_weight(dset_denom.lumi)
            lumi = dset_denom.lumi
        elif dset_denom.isMC:
            dset_denom.compute_weight(dset_num.lumi)
            lumi = dset_num.lumi
        else:
            print("[WARNING] Both datasets are data, not performing any lumi-weighting")
            if np.isclose(dset_num.lumi, dset_denom.lumi):
                lumi = dset_num.lumi
            else:
                lumi = None

    bins = np.asarray(spec['bins'])

    variable = parse_splt(spec['variable'])
    cut = parse_andcuts(spec['cut'])

    variable_pc = variable.to_pyarrow_expression()
    if variable_pc is None:
        raise ValueError("Failed to convert variable to PyArrow expression")
    
    cut_pc = cut.to_pyarrow_expression()

    wt_num_pc = wt_num.to_pyarrow_expression()
    wt_denom_pc = wt_denom.to_pyarrow_expression()
    if wt_num_pc is None or wt_denom_pc is None:
        raise ValueError("Failed to convert weights to PyArrow expressions")

    trial_smoothings = [ 
        SplineSmoothing(
            degree = 3, smoothing_factor = factor
        )
        for factor in spec['trial_smoothings']
    ]
    trial_smoothing_labels = [
        "Spline (s = %g)" % factor
        for factor in spec['trial_smoothings']
    ]

    compare_smoothings(
        dset_num=dset_num,
        dset_denom=dset_denom,
        cut=cut_pc,
        variable=variable_pc,
        wtvar_num=wt_num_pc,
        wtvar_denom=wt_denom_pc,
        bins=bins,
        smoothings=trial_smoothings,
        labels=trial_smoothing_labels,
        logx=spec['logx'],
        logy=spec['logy'],
        isMC=isMC,
        lumi=lumi,
        plot_path=os.path.join('reweighting/data/TEST-%s' % spec['name']),
        xlabel=spec['xlabel'],
        ylabel=spec['ylabel']
    )

    final_smoothings = [
        SplineSmoothing(
            degree = 3, smoothing_factor = spec['final_smoothing']
        )
    ]
    final_smoothing_labels = [
        'Spline'
    ]
    compare_smoothings(
        dset_num=dset_num,
        dset_denom=dset_denom,
        cut=cut_pc,
        variable=variable_pc,
        wtvar_num=wt_num_pc,
        wtvar_denom=wt_denom_pc,
        bins=bins,
        smoothings=final_smoothings,
        labels=final_smoothing_labels,
        logx=spec['logx'],
        logy=spec['logy'],
        isMC=isMC,
        lumi=lumi,
        plot_path=os.path.join('reweighting', 'data', spec['name']),
        xlabel=spec['xlabel'],  
        ylabel=spec['ylabel']
    )
    dump_smoothings(
        final_smoothings,
        final_smoothing_labels,
        varname = variable.key,
        vardesc = spec['variable_pcstr'],
        output_path = os.path.join('reweighting', 'data', spec['name'] + '.json')
    )