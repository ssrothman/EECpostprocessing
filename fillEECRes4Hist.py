import argparse

parser = argparse.ArgumentParser(description="Fill EEC Res4 Histograms")
parser.add_argument('Runtag', type=str, help='CMSSW Run Tag')
parser.add_argument('Sample', type=str, help='Sample Name')
parser.add_argument("Binner", type=str, help='Binner Name')
parser.add_argument('Hist', type=str, help='Histogram Name')
parser.add_argument('Objsyst', type=str, help='Object Systematic')
parser.add_argument('Wtsyst', type=str, help='Weight Systematic')
parser.add_argument("--nboot", type=int, default=100, help="Number of bootstrap samples")
parser.add_argument('--skipNominal', action='store_true', help="Skip nominal weight")
parser.add_argument("--prebinned", action='store_true', help="Data is pre-binned")
parser.add_argument("--nbatch", type=int, default=-1, help="Number of batches to process")
parser.add_argument("--rng", type=int, default=0, help="Random number generator seed offset")

args = parser.parse_args()

import os

basepath = f'/ceph/submit/data/group/cms/store/user/srothman/EEC/{args.Runtag}/{args.Sample}/{args.Binner}'
print(f"Basepath: {basepath}")
subpaths = os.scandir(basepath)

options = []
for subpath in subpaths:
    if not (subpath.is_dir()):
        continue
    options.append(subpath.name)

if (len(options) == 1):
    print()
    print("Only one option found: %s" % options[0])
    print("No user input needed :D")
    print()
    thepath = os.path.join(basepath, options[0])
else:
    print()
    print("Multiple options found:")
    for i, option in enumerate(options):
        print(f"{i}: {option}")
    print()
    choice = int(input("Please select a number: "))
    thepath = os.path.join(basepath, options[choice])
    print()

if args.Hist == 'wtratio':
    outpath = os.path.join(thepath, args.Objsyst, 'wtratio')
    thepath = os.path.join(thepath, args.Objsyst, 'transfer')
else:
    thepath = os.path.join(thepath, args.Objsyst, args.Hist)
    outpath = thepath

print("The path is: %s" % thepath)


from buildEECres4Hists import fill_hist_from_parquet, fill_transferhist_from_parquet, fill_wtratiohist_from_parquet

print("\nBuilding hist...")
if args.Hist == 'transfer':
    H = fill_transferhist_from_parquet(thepath, args.Wtsyst)
elif args.Hist == 'wtratio':
    H = fill_wtratiohist_from_parquet(thepath, args.Wtsyst)
else:
    H = fill_hist_from_parquet(thepath, args.nboot, 
                               args.Wtsyst, args.rng,
                               prebinned = args.prebinned,
                               nbatch = args.nbatch,
                               skipNominal = args.skipNominal)
print("Done.\n")


outfile = outpath
outfile += '_%s_%s' % (args.Objsyst, args.Wtsyst)
if args.nboot > 0:
    outfile += '_boot%d' % args.nboot
    outfile += '_rng%d' % args.rng
if args.skipNominal:
    outfile += '_skipNominal'
if args.nbatch > 0:
    outfile += '_first%d' % args.nbatch

outfile += '.pkl'

print("Saving result...")
print("Output file: %s" % outfile)
import pickle
with open(outfile, 'wb') as f:
    pickle.dump(H, f)
print("Done.\n")
