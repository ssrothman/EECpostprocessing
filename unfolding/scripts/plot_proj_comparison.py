#!/usr/bin/env python

import fasteigenpy as eigen

import argparse
from pathlib import Path
import numpy as np
import os.path

from simonplot import plot_histogram
from simonplot.binning import PrebinnedBinning
from simonplot.cut.PrebinnedCut import SliceOperation
from simonplot.variable import BasicPrebinnedVariable, ConstantVariable
from simonplot.variable.PrebinnedVariable import NormalizePerBlock
from simonplot.plottables.PrebinnedDatasets import ValCovPairDataset

from unfolding.histogram import Histogram, UnfoldedHistogram, get_genreco_name


def unfolded_to_histogram(uh: UnfoldedHistogram) -> Histogram:
    nGen = len(uh.baseline)
    x_gen = uh.x[:nGen]
    values = x_gen * uh.baseline
    invhess_gen = uh.invhess[:nGen, :nGen]
    d = uh.baseline
    covmat = d[:, None] * invhess_gen * d[None, :]
    return Histogram(values, covmat, uh.binning)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare unfolded projected EEC histograms per pT bin."
    )
    parser.add_argument(
        "histogram_paths",
        nargs="+",
        type=str,
        help="Path to saved unfolded histogram directories",
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        help="Path to the output folder where the plots will be saved.",
        default='./plots'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        type=str,
        help="Labels for the histograms.",
        default=None
    )
    parser.add_argument(
        '--extra-text',
        type=str,
        help="Additional text to include in the plots.",
        default=None
    )
    args = parser.parse_args()

    args.extra_text = args.extra_text.replace('\\n', '\n') if args.extra_text else None

    if args.labels is None:
        args.labels = [os.path.basename(p.rstrip('/')) for p in args.histogram_paths]

    histogram_dirs = [Path(p).expanduser().resolve() for p in args.histogram_paths]
    for histogram_dir in histogram_dirs:
        if not histogram_dir.is_dir():
            raise NotADirectoryError(f"Histogram path is not a directory: {histogram_dir}")

    unfolded = [UnfoldedHistogram.from_disk(str(d)) for d in histogram_dirs]
    histograms = [unfolded_to_histogram(uh) for uh in unfolded]

    axnames = histograms[0].binning.axis_names
    ptname = get_genreco_name(axnames, 'Jpt')
    ptedges = histograms[0].binning.axis_edges(ptname)

    datasets = [
        ValCovPairDataset(
            key=f'hist_{i}',
            color=None,
            label=args.labels[i],
            data=(h.values, h.covmat),
            binning=h.binning,
            isMC=True,
        )
        for i, h in enumerate(histograms)
    ]

    variable = NormalizePerBlock(BasicPrebinnedVariable(), [])
    weight   = ConstantVariable(1.0)
    binning  = PrebinnedBinning()

    os.makedirs(args.output_folder, exist_ok=True)

    for ipt in range(len(ptedges) - 1):
        lo, hi = ptedges[ipt], ptedges[ipt+1]
        if np.isinf(lo) or np.isinf(hi):
            continue
        cut = SliceOperation({ptname: (lo, hi)}, [])
        plot_histogram(
            variable, cut, weight, datasets, binning,
            extratext=args.extra_text,
            output_folder=args.output_folder,
            override_filename='comparison_pt%d' % int(lo),
            no_lumi_normalization=True,
            logx=True,
            logy=True,
        )


if __name__ == "__main__":
    main()
