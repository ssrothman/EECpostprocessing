from typing import List

import hist

from reweighting.smoothing import SmoothingProtocol
import simonplot as splt
import numpy as np
from simonplot.typing.Protocols import CutProtocol, VariableProtocol
from simonplot.util.evaluate import evaluate_on_dataset

import matplotlib.pyplot as plt



def compare_smothings(
    dset_num : splt.plottables.ParquetDataset,
    dset_denom : splt.plottables.ParquetDataset,
    cut : CutProtocol,
    variable : VariableProtocol,
    wtvar_num : VariableProtocol,
    wtvar_denom : VariableProtocol,
    bins : np.ndarray,
    smoothings : List[SmoothingProtocol | None],
    labels : List[str],
    logx : bool,
    logy : bool
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

    smoothed = []
    for smoothing in smoothings:
        if smoothing is None:
            smoothed.append(ratio)
        else:
            smoothed.append(smoothing(ratio, ratioerr, bins))

    fig = splt.util.common.setup_canvas()
    ax = splt.util.common.make_oneax(fig)
    splt.util.common.add_cms_legend(ax, False, None)

    for sm, label in zip(smoothed, labels):
        ax.plot(0.5 * (bins[:-1] + bins[1:]), sm, label=label)

    ax.errorbar(
        0.5 * (bins[:-1] + bins[1:]), ratio,
        yerr=ratioerr, xerr =0.5 * (bins[1:] - bins[:-1]),
        fmt='o', label='Ratio (no smoothing)', color='black'
    )
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.legend()
    plt.savefig('test.png')
    plt.show()

def compute_ratio(
    dset_num : splt.plottables.ParquetDataset,
    dset_denom : splt.plottables.ParquetDataset,
    cut : CutProtocol,
    variable : VariableProtocol,
    wtvar_num : VariableProtocol,
    wtvar_denom : VariableProtocol,
    bins : np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  
    
    print("evaluating numerator")
    vals_num = evaluate_on_dataset(
        dset_num, variable, cut # type: ignore
    )
    wt_num = evaluate_on_dataset(
        dset_num, wtvar_num, cut # type: ignore
    )
    print("evaluating denominator")
    vals_denom = evaluate_on_dataset(
        dset_denom, variable, cut # type: ignore
    )
    wt_denom = evaluate_on_dataset(
        dset_denom, wtvar_denom, cut # type: ignore
    )

    print("binning")
    H_num = hist.Hist(
        hist.axis.Variable(bins, overflow=False, underflow=False),
        storage=hist.storage.Weight(),
    )
    H_denom = hist.Hist(
        hist.axis.Variable(bins, overflow=False, underflow=False),
        storage=hist.storage.Weight(),
    )

    H_num.fill(vals_num, weight=wt_num)
    H_denom.fill(vals_denom, weight=wt_denom)

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