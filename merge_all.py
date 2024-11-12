import argparse

parser = argparse.ArgumentParser(description='Merge all files in a directory')

parser.add_argument('path', type=str, help='Path to the directory')

args = parser.parse_args()

print(args.path)

import os
files = os.listdir(args.path)

from iadd import iadd
import pickle
from tqdm import tqdm
result = None
for file in tqdm(files):
    if file.endswith('.pkl') and file != 'merged.pkl':
        with open(os.path.join(args.path, file), 'rb') as f:
            if result is None:
                result = pickle.load(f)
            else:
                result = iadd(result, pickle.load(f))

with open(os.path.join(args.path, 'merged.pkl'), 'wb') as f:
    pickle.dump(result, f)

