from typing import List, Sequence

import hist

from reweighting.smoothing import SmoothingProtocol
import simonplot as splt
import numpy as np
from simonplot.typing.Protocols import CutProtocol, VariableProtocol
from simonplot.util.evaluate import evaluate_on_dataset
from simonplot.util.common import setup_canvas, make_oneax, add_cms_legend, savefig
import pyarrow.dataset as ds

import matplotlib.pyplot as plt

def dump_smoothings(smoothing_l : Sequence[SmoothingProtocol], names : Sequence[str], varname : str, output_path : str, vardesc : str):
    corrections = [smoothing.to_correctionlib(name, varname, vardesc) for smoothing, name in zip(smoothing_l, names)]
    from correctionlib import schemav2
    cset = schemav2.CorrectionSet(
        schema_version = 2,
        corrections = corrections
    )
    with open(output_path, 'w') as f:
        f.write(cset.model_dump_json(indent=2))

def compare_smoothings(
    dset_num : splt.plottables.ParquetDataset | splt.plottables.DatasetStack,
    dset_denom : splt.plottables.ParquetDataset | splt.plottables.DatasetStack,
    cut : ds.Expression | None,
    variable : ds.Expression,
    wtvar_num : ds.Expression | None,
    wtvar_denom : ds.Expression | None,
    bins : np.ndarray,
    smoothings : Sequence[SmoothingProtocol],
    labels : List[str],
    logx : bool,
    logy : bool,
    isMC : bool,
    lumi : float | None,
    plot_path : str ,
    ylabel : str,
    xlabel : str
):  
    ratio, ratioerr = compute_ratio(
        dset_num,
        dset_denom,
        cut,
        variable,
        wtvar_num,
        wtvar_denom,
        bins
    )

    for smoothing in smoothings:
        smoothing(ratio, ratioerr, bins)

    fig = setup_canvas()
    ax = make_oneax(fig)
    add_cms_legend(ax, not isMC, lumi)

    smoothx = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 1000, base=10)
    for sm, label in zip(smoothings, labels):
        ax.plot(smoothx, sm.evaluate(smoothx), label=label)

    ax.errorbar(
        0.5 * (bins[:-1] + bins[1:]), ratio,
        yerr=ratioerr, xerr =0.5 * (bins[1:] - bins[:-1]),
        fmt='o', label='Ratio', color='black'
    )
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.legend()

    #tight layout
    fig.tight_layout()

    savefig(fig, plot_path)
    plt.close(fig)

def compute_ratio(
    dset_num : splt.plottables.ParquetDataset | splt.plottables.DatasetStack,
    dset_denom : splt.plottables.ParquetDataset | splt.plottables.DatasetStack,
    cut : ds.Expression | None,
    variable : ds.Expression,
    wtvar_num : ds.Expression | None,
    wtvar_denom : ds.Expression | None,
    bins : np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  
    
    print("evaluating numerator")
    H_num = hist.Hist(
        hist.axis.Variable(bins, overflow=False, underflow=False, name='x'),
        storage=hist.storage.Weight()
    )
    dset_num.streaming_fill_histogram(
        H_num, 
        {'x': variable}, wtvar_num, cut
    )
    print("evaluating denominator")
    H_denom = hist.Hist(
        hist.axis.Variable(bins, overflow=False, underflow=False, name='x'),
        storage=hist.storage.Weight()
    )
    dset_denom.streaming_fill_histogram(
        H_denom,
        {'x': variable}, wtvar_denom, cut
    )

    num = H_num.values(flow=True)
    denom = H_denom.values(flow=True)

    numerr = np.sqrt(np.asarray(H_num.variances(flow=True)))
    denomerr  = np.sqrt(np.asarray(H_denom.variances(flow=True)))

    ratio = num / denom
    ratioerr = np.sqrt(
        np.square(numerr / denom) + 
        np.square(num * denomerr / np.square(denom))
    )

    return ratio, ratioerr