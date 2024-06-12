import pickle
import os
import argparse
import glob
from coffea.processor.accumulator import iadd

parser = argparse.ArgumentParser(description='Merge data from different eras')
parser.add_argument('binning', type=str, help='Name of binning subfolder')
parser.add_argument('--flags', type=str, nargs="*", default = [])
parser.add_argument('--extra_flags', type=str, nargs="*", default = [])

args = parser.parse_args()

from samples.latest import SAMPLE_LIST

H = None
for era in ['DATA_2018A', 'DATA_2018B', 'DATA_2018C', 'DATA_2018D']:
    print(era)

    Hnext = SAMPLE_LIST.get_hist(era, args.binning, args.flags)

    if H is None:
        H = Hnext
    else:
        iadd(H, Hnext)

    del Hnext

print("sumwt =", H['sumwt'])
destpath = SAMPLE_LIST.get_basepath('DATA_2018UL', args.binning)
os.makedirs(destpath, exist_ok=True)
fname = 'hist'
for flag in args.flags + args.extra_flags:
    fname += '_'+flag
fname += '.pkl'
print("Writing to", destpath)
with open(os.path.join(destpath, fname), 'wb') as f:
    pickle.dump(H, f)
