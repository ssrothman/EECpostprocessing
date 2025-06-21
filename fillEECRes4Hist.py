import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Fill EEC Res4 Histograms")
parser.add_argument('Runtag', type=str, help='CMSSW Run Tag', default=None, nargs='?')
parser.add_argument('Sample', type=str, help='Sample Name', default=None, nargs='?')
parser.add_argument("Binner", type=str, help='Binner Name', default=None, nargs='?')
parser.add_argument('Hist', type=str, help='Histogram Name', default=None, nargs='?')
parser.add_argument('Objsyst', type=str, help='Object Systematic', default=None, nargs='?')
parser.add_argument('Wtsyst', type=str, help='Weight Systematic', default=None, nargs='?')

parser.add_argument('--usexrootd', action='store_true')
parser.add_argument('--thepath', type=str, default=None)
parser.add_argument('--outfile', type=str, default=None)
parser.add_argument('--passedwtsyst', type=str, default=None)

parser.add_argument("--nboot", type=int, default=100, help="Number of bootstrap samples")
parser.add_argument("--rng", type=int, default=0, help="Random number generator seed offset")
parser.add_argument('--skipNominal', action='store_true', help="Skip nominal weight")

parser.add_argument("--prebinned", action='store_true', help="Data is pre-binned")

parser.add_argument("--nbatch", type=int, default=-1, help="Number of batches to process")

parser.add_argument('--statN', type=int, default=-1, help="Number of statistically-independent splits to create")
parser.add_argument('--statK', type=int, default=-1, help="Which statistical split we are processing (k in [0, statN-1])")

parser.add_argument('--kinreweight_path', type=str, default=None, help="Path to kinreweight correctionlib json")
parser.add_argument('--kinreweight_key', type=str, default=None, help="Key in kinreweight correctionlib json")

parser.add_argument("--slurm", action='store_true')

args = parser.parse_args()

import sys
from shlex import quote
print(' '.join(quote(s) for s in sys.argv))
if args.Hist == 'transfer' and args.nboot > 0:
    print("\tno transfer bootstrap")
    sys.exit(0)

if args.usexrootd and args.thepath is None:
    raise ValueError("If using xrootd, you must specify the path with --thepath")

if args.thepath is not None:
    if args.Binner is not None or args.Sample is not None or args.Hist is not None or args.Objsyst is not None or args.Wtsyst is not None:
        raise ValueError("If using --thepath, you cannot specify Binner, Sample, Hist, Objsyst or Wtsyst")
else:
    if args.Binner is None or args.Sample is None or args.Hist is None or args.Objsyst is None or args.Wtsyst is None:
        raise ValueError("You must specify Binner, Sample, Hist, Objsyst and Wtsyst if not using --thepath")

if args.thepath is not None and args.outfile is None:
    raise ValueError("If using --thepath, you must specify --outfile")

if args.passedwtsyst is not None:
    if args.Wtsyst is not None:
        raise ValueError("You cannot specify both Wtsyst and passedwtsyst")

    args.Wtsyst = args.passedwtsyst

if args.Wtsyst is None:
    raise ValueError("You must specify Wtsyst, either positionally or with --passedwtsyst")

import os

if args.usexrootd:
    from fsspec_xrootd import XRootDFileSystem
    fs = XRootDFileSystem(hostid='submit55.mit.edu', 
                          timeout=30)
else:
    fs = None
    
if args.thepath is None:
    basepath = f'/ceph/submit/data/group/cms/store/user/srothman/EEC/{args.Runtag}/{args.Sample}/{args.Binner}'
    #print(f"Basepath: {basepath}")
    subpaths = os.scandir(basepath)

    options = []
    for subpath in subpaths:
        if not (subpath.is_dir()):
            continue
        if not (subpath.name.startswith('hists')):
            continue

        options.append(subpath.name)

    if (len(options) == 1):
        #print()
        #print("Only one option found: %s" % options[0])
        #print("No user input needed :D")
        #print()
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

    thepath = os.path.join(thepath, args.Objsyst, args.Hist)

    #print("The path is: %s" % thepath)
else:
    thepath = args.thepath


import hashlib
pathhash = hashlib.md5(basepath.encode())
#first 32 bits as int
pathhash = np.uint32(int(pathhash.hexdigest()[:8], 16))
args.rng = args.rng + pathhash

if args.outfile is None:
    outfile = thepath
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
    if args.kinreweight_path is not None:
        outfile += '_kinreweight'

    outfile += '.pkl'
else:
    outfile = args.outfile

if (args.usexrootd and fs.exists(outfile)) or (not args.usexrootd and os.path.exists(outfile)):
    #print("destination already exists! skipping")
    import sys
    sys.exit(1)
else:
    print("Trying to produce output file: %s" % outfile)

if args.slurm:
    import subprocess
    import sys
    from shlex import quote
    
    with open("slurm_template.txt", 'r') as f:
        slurm_script = f.read()

    command = ' '.join(quote(s) for s in sys.argv)
    command = command.replace('--slurm', '')
    #pipe stderr to stdout
    command += ' 2>&1'
    slurm_script = slurm_script.replace("COMMAND", "python " + command)
    print(slurm_script)

    import random
    uuid = random.getrandbits(64)
    uuidstr = '%016x' % uuid
    uuidstr = "%s_%s_%s_"%(args.Sample, args.Hist, args.Wtsyst) + uuidstr

    slurm_script = slurm_script.replace("UUID", uuidstr)

    desired_mem = '8g' if args.Hist == 'transfer' else '8g'
    slurm_script = slurm_script.replace('MEM', desired_mem)

    with open("slurm/submit_%s.sh" % uuidstr, 'w') as f:
        f.write(slurm_script)
    print("Submitting job to SLURM...")
    q = subprocess.run(['sbatch', 'slurm/submit_%s.sh' % uuidstr])
    print(q)
    print("Job submitted with UUID:", uuidstr)
    print("Exiting.")
    import time
    sys.exit(0)

from buildEECres4Hists import fill_hist_from_parquet, fill_transferhist_from_parquet

if args.kinreweight_path is not None:
    from correctionlib import CorrectionSet
    cset = CorrectionSet.from_file(args.kinreweight_path)
    kinreweight_func = cset[args.kinreweight_key].evaluate
else:
    kinreweight_func = None

print("\nBuilding hist...")
if args.Hist == 'transfer':
    print("Clipping bootstrap -> 0")
    H = fill_transferhist_from_parquet(thepath, 
                                       bootstrap = 0, 
                                       systwt = args.Wtsyst, 
                                       random_seed = args.rng,
                                       skipNominal = args.skipNominal,
                                       statN = args.statN,
                                       statK = args.statK,
                                       kinreweight = kinreweight_func,
                                       fs = fs)
else:
    H = fill_hist_from_parquet(thepath, args.nboot, 
                               args.Wtsyst, args.rng,
                               prebinned = args.prebinned,
                               nbatch = args.nbatch,
                               skipNominal = args.skipNominal,
                               statN = args.statN,
                               statK = args.statK,
                               kinreweight = kinreweight_func,
                               fs = fs)
print("Done.\n")


print("Saving result...")
print("Output file: %s" % outfile)
import pickle
with open(outfile, 'wb') as f:
    pickle.dump(H, f)
print("Done.\n")
