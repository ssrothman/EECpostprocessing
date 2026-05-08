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
parser.add_argument('--rebinning-reco', type=str, default=None)
parser.add_argument('--rebinning-gen', type=str, default=None)
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
        
        Hreco = Histogram.from_dataset(
            datacfg['dset'],
            datacfg['hist'],
            'totalReco',
            rebinning = args.rebinning_reco
        )
        Hreco.compute_invcov() # precompute the inverse covariance matrix for the reco histogram

        Hreco.dump_to_disk('%s_reco'%datacfg['name'])

        if datacfg['dset']['isMC']:
            Hgen = Histogram.from_dataset(
                datacfg['dset'],
                datacfg['hist'],
                'totalGen',
                rebinning = args.rebinning_gen
            )
            Hgen.compute_invcov() # precompute the inverse covariance matrix for the gen histogram
            Hgen.compute_sqrt() # precompute the sqrt of the covariance matrix for the gen
                                # used for bootstrapping the forward covariance matrix

            Hgen.dump_to_disk('%s_gen'%datacfg['name'])

if not args.skip_model and not (args.only is not None and 'model' not in args.only):
    model = DetectorModel.from_dataset(
        cfg['model'],
        rebinning_reco = args.rebinning_reco,
        rebinning_gen = args.rebinning_gen
    )
    model.dump_to_disk('model')

    mcgen = Histogram.from_dataset(
        cfg['model']['dset'],
        {
            'wtsyst' : 'nominal',
            'objsyst' : 'nominal'
        },
        'totalGen',
        rebinning = args.rebinning_gen
    )
    mcgen.dump_to_disk('mcgen')