#!/usr/bin/env python

import argparse
import fasteigenpy as eigen

parser = argparse.ArgumentParser(description="Compute good x0 guess for unfolding")
parser.add_argument("reco", type=str, help="Path to reco histogram")
parser.add_argument("--baseline", type=str, default = 'mcgen', help="Path to baseline histogram")
parser.add_argument("--model", type=str, default = 'model', help="Path to detector model")

args = parser.parse_args()

from unfolding.histogram import Histogram
from unfolding.detectormodel import DetectorModel
import numpy as np
import os.path

reco = Histogram.from_disk(args.reco)
baseline = Histogram.from_disk(args.baseline)
model = DetectorModel.from_disk(args.model)

beta0 = model.getGoodX0(reco.values, baseline.values)

beta0path = os.path.join(args.reco, 'goodBeta0.npy')
print("Saving to", beta0path)
np.save(beta0path, beta0)