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

Hreco.dump_to_disk('reco')

if cfg['data']['dset']['isMC']:
    Hgen = Histogram.from_dataset(
        cfg['data']['dset'],
        cfg['data']['hist'],
        'totalGen'
    )
    Hgen.dump_to_disk('gen')

model = DetectorModel.from_dataset(
    cfg['model']
)
model.dump_to_disk('model')