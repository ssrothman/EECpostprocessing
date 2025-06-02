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
parser.add_argument('--statN', type=int, default=-1, help="Number of statistically-independent splits to create")
parser.add_argument('--statK', type=int, default=-1, help="Which statistical split we are processing (k in [0, statN-1])")
parser.add_argument("--rng", type=int, default=0, help="Random number generator seed offset")
parser.add_argument("--slurm", action='store_true')

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

outfile = outpath
outfile += '_%s_%s' % (args.Objsyst, args.Wtsyst)
if args.nboot > 0:
    outfile += '_boot%d' % args.nboot
    outfile += '_rng%d' % args.rng
if args.skipNominal:
    outfile += '_skipNominal'
if args.nbatch > 0:
    outfile += '_first%d' % args.nbatch
if args.statN > 0:
    outfile += '_%dstat%d' % (args.statN, args.statK)

outfile += '.pkl'

if os.path.exists(outfile):
    print("destination already exists! skipping")
    import sys
    sys.exit(1)

if args.slurm:
    import subprocess
    import sys
    from shlex import quote
    
    with open("slurm_template.txt", 'r') as f:
        slurm_script = f.read()

    command = ' '.join(quote(s) for s in sys.argv)
    command = command.replace('--slurm', '')
    slurm_script = slurm_script.replace("COMMAND", "python " + command)
    print(slurm_script)

    import random
    uuid = random.getrandbits(64)
    uuidstr = '%016x' % uuid
    uuidstr = "%s_%s_%s_"%(args.Sample, args.Hist, args.Wtsyst) + uuidstr

    slurm_script = slurm_script.replace("UUID", uuidstr)

    with open("slurm/submit_%s.sh" % uuidstr, 'w') as f:
        f.write(slurm_script)
    print("Submitting job to SLURM...")
    q = subprocess.run(['sbatch', 'slurm/submit_%s.sh' % uuidstr])
    print(q)
    print("Job submitted with UUID:", uuidstr)
    print("Exiting.")
    import time
    time.sleep(1)
    sys.exit(0)

from buildEECres4Hists import fill_hist_from_parquet, fill_transferhist_from_parquet, fill_wtratiohist_from_parquet

print("\nBuilding hist...")
if args.Hist == 'transfer':
    print("Clipping bootstrap -> 0")
    H = fill_transferhist_from_parquet(thepath, 
                                       bootstrap = 0, 
                                       systwt = args.Wtsyst, 
                                       random_seed = args.rng,
                                       skipNominal = args.skipNominal,
                                       statN = args.statN,
                                       statK = args.statK)
elif args.Hist == 'wtratio':
    H = fill_wtratiohist_from_parquet(thepath, args.Wtsyst, args.statN, args.statK)
else:
    H = fill_hist_from_parquet(thepath, args.nboot, 
                               args.Wtsyst, args.rng,
                               prebinned = args.prebinned,
                               nbatch = args.nbatch,
                               skipNominal = args.skipNominal,
                               statN = args.statN,
                               statK = args.statK)
print("Done.\n")


print("Saving result...")
print("Output file: %s" % outfile)
import pickle
with open(outfile, 'wb') as f:
    pickle.dump(H, f)
print("Done.\n")
