import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("skimmer", type=str)

args = parser.parse_args()

import os
import os.path
basepath = "/ceph/submit/data/user/s/srothman/EEC/"
path = os.path.join(basepath, args.tag, args.dataset, args.skimmer)
options = os.listdir(path)

real_options = []

for option in options:
    if option.endswith('.pickle'):
        real_options.append(option)

if len(real_options) == 1:
    print("only one option. No user input needed")
    thefile = real_options[0]
else:
    print("%d options found"%len(real_options))
    for i, option in enumerate(real_options):
        print("\t", i, option)
    choice = input("Please choose an option: ")
    choice = int(choice)
    thefile = real_options[choice]

with open(os.path.join(path, thefile), 'rb') as f:
    data = pickle.load(f)

print()

if 'errd' in data:
    print("There were %d failed files"%len(data['errd']))
else:
    print("Everything ran successfully")

