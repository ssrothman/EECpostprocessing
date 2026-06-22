#!/usr/bin/env python

import json
import fasteigenpy as eigen
from unfolding.histogram import Histogram
from unfolding.specs import unfoldingworkspacespec
from unfolding.detectormodel import DetectorModel

import argparse

parser = argparse.ArgumentParser(description='Setup the unfolding workspace by dumping the histograms and model to disk')
parser.add_argument('--skip-model', action='store_true', help='Skip setting up the detector model')
parser.add_argument('--skip-data', action='store_true', help='Skip setting up the data histograms')
parser.add_argument('--only', type=str, nargs='+', default=None, help='Only setup the specified quantities (by name)')
parser.add_argument('--rebinning-reco', type=str, default='rebinning_reco.json', help='Optional rebinning config for reco histograms')
parser.add_argument('--rebinning-gen', type=str, default='rebinning_gen.json', help='Optional rebinning config for gen histograms')
parser.add_argument('--rebin', action='store_true', help='Whether to apply rebinning to the histograms (if false, the rebinning configs are ignored)')
parser.add_argument('--skip-invcov', action='store_true', help='Skip precomputing the inverse covariance matrices for the histograms')
args = parser.parse_args()

with open("config.json", 'r') as f:
    cfg : unfoldingworkspacespec = json.load(f)

if not args.skip_data:
    if isinstance(cfg['data'], dict):
        datalist = [cfg['data']]
    elif isinstance(cfg['data'], list):
        datalist = cfg['data']
    else:
        raise ValueError("cfg['data'] must be either a dict or a list of dicts")

    for datacfg in datalist:
        if args.only is not None and datacfg['name'] not in args.only:
            print(f"Skipping {datacfg['name']} since it's not in the --only list")
            continue
        else:
            print("Setting up data histograms for", datacfg['name'])

        Hreco = Histogram.from_dataset(
            datacfg['dset'],
            datacfg['hist'],
            'totalReco',
            rebinning = args.rebinning_reco if args.rebin else None,
            nocov = datacfg.get('nocov', False)
        )
        if not args.skip_invcov and not datacfg.get('nocov', False):
            Hreco.compute_invcov() # precompute the inverse covariance matrix for the reco histogram

        Hreco.dump_to_disk('%s_reco'%datacfg['name'])

        if datacfg['dset']['isMC'] and not datacfg.get('skipgen', False):
            Hgen = Histogram.from_dataset(
                datacfg['dset'],
                datacfg['hist'],
                'totalGen',
                rebinning = args.rebinning_gen if args.rebin else None,
                nocov = datacfg.get('nocov', False)
            )
            if not args.skip_invcov and not datacfg.get('nocov', False):
                Hgen.compute_invcov() # precompute the inverse covariance matrix for the gen histogram
                Hgen.compute_sqrt() # precompute the sqrt of the covariance matrix for the gen
                                    # used for bootstrapping the forward covariance matrix

            Hgen.dump_to_disk('%s_gen'%datacfg['name'])

if not args.skip_model and not (args.only is not None and 'model' not in args.only):
    print("Setting up detecor model")
    model = DetectorModel.from_dataset(
        cfg['model'],
        rebinning_reco = args.rebinning_reco if args.rebin else None, 
        rebinning_gen = args.rebinning_gen if args.rebin else None
    )
    model.dump_to_disk('model')

    if 'mcgen' in cfg['model']:
        mcgen_dset = cfg['model']['mcgen']
    else:
        mcgen_dset = {
            'dset' : cfg['model']['dset'],
            'hist' : {
                'wtsyst' : 'nominal',
                'objsyst' : 'nominal'
            }
        }

    mcgen = Histogram.from_dataset(
        mcgen_dset['dset'],
        mcgen_dset['hist'],
        'totalGen',
        rebinning = args.rebinning_gen if args.rebin else None,
        nocov = mcgen_dset.get('nocov', False)
    )
    mcgen.dump_to_disk('mcgen')