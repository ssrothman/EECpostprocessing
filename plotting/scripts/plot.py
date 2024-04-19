import argparse
import os

parser = argparse.ArgumentParser(description='Plotting script')
parser.add_argument('which', type=str, help='Which plots to make')
parser.add_argument('--folder_suffix', type=str, help='Folder to save plots in', default='')
parser.add_argument('--show', action='store_true', help='Show the plots')
parser.add_argument('--tags', type=str, nargs='*', help='Dataset tags')

args = parser.parse_args()

from samples.latest import SAMPLE_LIST
from plotting.scripts.HTvalidation import doHTval
from plotting.scripts.kinDataMC import doKinDataMC

folder = os.path.join('plots', SAMPLE_LIST.tag+args.folder_suffix)
show = args.show

print(args.show)

if args.which == "HTval":
    doHTval(os.path.join(folder, "HTval"), show)
elif args.which == "KinDataMC":
    doKinDataMC(os.path.join(folder, "KinDataMC"), show)
else:
    print("Unknown option")
    exit(1)
