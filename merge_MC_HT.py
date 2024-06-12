import pickle
import os
import argparse
import glob
import json
from coffea.processor.accumulator import iadd

parser = argparse.ArgumentParser(description='Merge data from different eras')
parser.add_argument('binning', type=str, help='Name of binning subfolder')
parser.add_argument('--flags', type=str, nargs="*", default = [])
parser.add_argument('--extra_flags', type=str, nargs="*", default = [])

args = parser.parse_args()

print(args.flags)

names = [
    'DYJetsToLL_HT-0to70',
    'DYJetsToLL_HT-70to100',
    'DYJetsToLL_HT-100to200',
    'DYJetsToLL_HT-200to400',
    'DYJetsToLL_HT-400to600',
    'DYJetsToLL_HT-600to800',
    'DYJetsToLL_HT-800to1200',
    'DYJetsToLL_HT-1200to2500',
    'DYJetsToLL_HT-2500toInf',
]

from samples.latest import SAMPLE_LIST

with open("configs/base.json") as f:
    config = json.load(f)

xsecs = config['xsecs']

base_xsec = xsecs['DYJetsToLL']
lumi = config['totalLumi']

def recursive_mult(d, factor):
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_mult(v, factor)
        else:
            d[k] *= factor

H = None
for name in names:
    print(name)
    print("\txsec:", xsecs[name])
    print("\tbase_xsec:", base_xsec)
    print("\tratio:", xsecs[name] / base_xsec)

    Hnext = SAMPLE_LIST.get_hist(name, args.binning, args.flags)

    factor = xsecs[name] / base_xsec / Hnext['sumwt']
    print("\tsumwt:", Hnext['sumwt'])
    print("\tfactor:", factor)
    recursive_mult(Hnext, factor)
    #print("\tsumwt =", Hnext['sumwt'])
    #print("\tsumwt_pass =", Hnext['sumwt_pass'])
    #print("\tsumwt_pass * lumi * base_sec / sumwt * 1000", Hnext['sumwt_pass'] * lumi * base_xsec / Hnext['sumwt'] * 1000)

    #new = vals * lumi * xseces[name] / sumwts * 100 

    if H is None:
        H = Hnext
    else:
        iadd(H, Hnext)

    del Hnext

print("sumwt =", H['sumwt'])
destpath = SAMPLE_LIST.get_basepath('DYJetsToLL_allHT', args.binning)
os.makedirs(destpath, exist_ok=True)
print("Writing to", destpath)
fname = 'hists'
for flag in args.flags + args.extra_flags:
    fname += '_' + flag
fname += '.pkl'
with open(os.path.join(destpath, fname), 'wb') as f:
    pickle.dump(H, f)
