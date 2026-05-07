#!/usr/bin/env python

import json
import fasteigenpy as eigen
from unfolding.histogram import Histogram
from unfolding.specs import unfoldingworkspacespec
from unfolding.detectormodel import DetectorModel

with open("config.json", 'r') as f:
    cfg : unfoldingworkspacespec = json.load(f)

Hreco = Histogram.from_dataset(
    cfg['data']['dset'],
    cfg['data']['hist'],
    'totalReco'
)
Hreco.compute_invcov() # precompute the inverse covariance matrix for the reco histogram

Hreco.dump_to_disk('reco')

if cfg['data']['dset']['isMC']:
    Hgen = Histogram.from_dataset(
        cfg['data']['dset'],
        cfg['data']['hist'],
        'totalGen'
    )
    Hgen.compute_invcov() # precompute the inverse covariance matrix for the gen histogram
    Hgen.compute_sqrt() # precompute the sqrt of the covariance matrix for the gen
                        # used for bootstrapping the forward covariance matrix

    Hgen.dump_to_disk('gen')


model = DetectorModel.from_dataset(
    cfg['model']
)
model.dump_to_disk('model')

mcgen = Histogram.from_dataset(
    cfg['model']['dset'],
    {
        'wtsyst' : 'nominal',
        'objsyst' : 'nominal'
    },
    'totalGen'
)
mcgen.dump_to_disk('mcgen')