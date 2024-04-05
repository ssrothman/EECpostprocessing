import pyarrow.dataset as ds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hist

# Load the dataset
data = ds.dataset('/data/submit/srothman/trainingdata', format='parquet').to_table().to_pandas()

passB = data['CvB'] 

H = hist.Hist(
    hist.axis.Regular(50, 0, 1, label="CvL", name="CvL",
                      underflow=False, overflow=False),
    hist.axis.Regular(50, 0, 1, label="CvB", name="CvB",
                      underflow=False, overflow=False),
    hist.axis.Variable([30, 80, 150, 1000], name = 'pt', label = 'pT',
                       underflow=False, overflow=False),
    hist.axis.IntCategory([0, 4, 5], name="genflav", label="flavor"),
)

mask = data['pt'] > 30
H.fill(CvL=data['CvL'][mask], 
       CvB=data['CvB'][mask], 
       genflav=data['hadronFlavour'][mask],
       pt=data['pt'][mask])

vals = H.values()

import plotting.plotEvent
plotting.plotEvent.taggerdists(H)
