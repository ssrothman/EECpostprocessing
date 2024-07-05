import argparse
import os.path

parser = argparse.ArgumentParser(description='Convert pkl to npy')

parser.add_argument('input', type=str, help='input pkl file')

args = parser.parse_args()

import pickle
import numpy as np

with open(args.input, 'rb') as f:
    data = pickle.load(f)

folder = os.path.dirname(args.input)
filename = os.path.basename(args.input)
prefix = os.path.splitext(filename)[0]

npyfolder = os.path.join(folder, prefix)

statsplit_idx = prefix.find("statsplit")

os.makedirs(os.path.join(folder, prefix), exist_ok=True)

if statsplit_idx < 0:
    for key, value in data.items():
        np.save(os.path.join(folder, prefix,  key + '.npy'), value)
else:
    N = int(prefix[statsplit_idx+9])
    for k in range(N):
        os.makedirs(os.path.join(folder, prefix, 'stat%d'%k), exist_ok=True)
        for key, value in data.items():
            if hasattr(value, '__len__'):
                np.save(os.path.join(folder, prefix, 'stat%d'%k, key + '.npy'), value[k])
    for key, value in data.items():
        if type(value) is not float:
            np.save(os.path.join(folder, prefix, key + '.npy'), np.sum(value, axis=0))
        else:
            np.save(os.path.join(folder, prefix, key + '.npy'), value)
