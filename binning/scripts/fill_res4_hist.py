import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Fill EEC Res4 Histograms")
parser.add_argument('Runtag', type=str, help='CMSSW Run Tag', nargs='?')
parser.add_argument('Sample', type=str, help='Sample Name', nargs='?')
parser.add_argument("Binner", type=str, help='Binner Name', nargs='?')
parser.add_argument('Hist', type=str, help='Histogram Name', nargs='?')
parser.add_argument('Objsyst', type=str, help='Object Systematic', nargs='?')
parser.add_argument('Wtsyst', type=str, help='Weight Systematic', nargs='?')

parser.add_argument('--usexrootd', action='store_true')

parser.add_argument("--nboot", type=int, default=0, help="Number of bootstrap samples")
parser.add_argument("--rng", type=int, default=0, help="Random number generator seed offset")
parser.add_argument('--skipNominal', action='store_true', help="Skip nominal weight")

parser.add_argument("--prebinned", action='store_true', help="Data is pre-binned")

parser.add_argument("--nbatch", type=int, default=-1, help="Number of batches to process")

parser.add_argument('--r123type', type=str, default='philox', 
                    choices=['philox', 'threefry'], 
                    help="Type of random number generator to use")

parser.add_argument('--collect_debug_info', action='store_true', 
                    help="Collect debug information for bootstrap weights")

parser.add_argument('--statN', type=int, default=-1, help="Number of statistically-independent splits to create")
parser.add_argument('--statK', type=int, default=-1, help="Which statistical split we are processing (k in [0, statN-1])")

parser.add_argument('--kinreweight_path', type=str, default=None, help="Path to kinreweight correctionlib json")
parser.add_argument('--kinreweight_key', type=str, default=None, help="Key in kinreweight correctionlib json")

exec_group = parser.add_mutually_exclusive_group(required=False)
exec_group.add_argument("--slurm", action='store_true')
exec_group.add_argument("--condor", action='store_true')

parser.add_argument('--force', action='store_true')

parser.add_argument("--mute", action='store_true',)

args = parser.parse_args()

if args.kinreweight_key == 'None':
    args.kinreweight_key = None
if args.kinreweight_path == 'None':
    args.kinreweight_path = None

if args.mute:
    import os
    import sys
    sys.stdout = open(os.devnull, 'w')

import sys
from shlex import quote
print(' '.join(quote(s) for s in sys.argv))
if args.Hist == 'transfer' and args.nboot > 0:
    print("\tno transfer bootstrap")
    sys.exit(0)

if args.Hist.startswith('directcov') and args.nboot > 0:
    print("\tno directcov bootstrap")
    sys.exit(0)

if args.usexrootd:
    from fsspec_xrootd import XRootDFileSystem
    fs = XRootDFileSystem(hostid='submit55.mit.edu', 
                          timeout=120)
else:
    from pyarrow.fs import LocalFileSystem
    from fsspec.implementations.arrow import ArrowFSWrapper
    fs = ArrowFSWrapper(LocalFileSystem())
    
basepath = f'/ceph/submit/data/group/cms/store/user/srothman/EEC/{args.Runtag}/{args.Sample}/{args.Binner}'
basepath_xrd = f'/store/user/srothman/EEC/{args.Runtag}/{args.Sample}/{args.Binner}'

#print(f"Basepath: {basepath}")
if args.usexrootd:
    subpaths = fs.listdir(basepath_xrd)
else:
    subpaths = fs.listdir(basepath)

options = []
for subpath in subpaths:
    if subpath['type'] != 'directory':
        continue
    if not (os.path.basename(subpath['name']).startswith('hists')):
        continue

    options.append(subpath['name'])

if (len(options) == 1):
    #print()
    #print("Only one option found: %s" % options[0])
    #print("No user input needed :D")
    #print()
    thepath = os.path.join(basepath, options[0])
else:
    if args.usexrootd:
        print("ERROR MULTIPLE OPTIONS")
        for option in options:
            print("\t", option)
        sys.exit(99)

    print()
    print("Multiple options found:")
    for i, option in enumerate(options):
        print(f"{i}: {option}")
    print()
    choice = int(input("Please select a number: "))
    thepath = os.path.join(basepath, options[choice])
    print()

if args.Hist.startswith('directcov'):
    whathist = args.Hist.split('_')[1]
    thepath = os.path.join(thepath, args.Objsyst, whathist)
else:
    thepath = os.path.join(thepath, args.Objsyst, args.Hist)

#print("The path is: %s" % thepath)

import hashlib
pathhash = hashlib.md5(basepath.encode())
#first 32 bits as int
pathhash = np.uint32(int(pathhash.hexdigest()[:8], 16))
args.rng = args.nboot * args.rng + pathhash

outfile = thepath
if args.Hist.startswith('directcov'):
    outfile = os.path.join(os.path.dirname(outfile), 
                           'directcov_' + os.path.basename(outfile))
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
    outfile += '_%s' % args.kinreweight_key
if args.prebinned:
    outfile += '_prebinned'
outfile += '_%s' % args.r123type
if args.collect_debug_info:
    outfile += '_debug'

outfile += '.pkl'

if fs.exists(outfile) and fs.stat(outfile)['size'] == 0:
    fs.rm_file(outfile)

if fs.exists(outfile) and not args.force:
    print("destination already exists! skipping")
    print(outfile)
    import sys
    sys.exit(0)
else:
    print("Trying to produce output file: %s" % outfile)

if args.slurm or args.condor:
    import subprocess
    import sys
    from shlex import quote
    command = ' '.join(quote(s) for s in sys.argv)
    #pipe stderr to stdout
    command += ' 2>&1'

    import random
    uuid = random.getrandbits(64)
    uuidstr = '%016x' % uuid
    uuidstr = "%s_%s_%s_%s_boot%d_rng%d_%dstat%d_"%(args.Sample, args.Hist, args.Objsyst, args.Wtsyst, args.nboot, args.rng, args.statN, args.statK) + uuidstr

    if args.slurm:
        with open("templates/slurm_template.txt", 'r') as f:
            slurm_script = f.read()

        command = command.replace('--slurm', '')
        slurm_script = slurm_script.replace("COMMAND", "python " + command)
        print(slurm_script)

        slurm_script = slurm_script.replace("UUID", uuidstr)

        desired_mem = '8g' if args.Hist == 'transfer' or 'directcov' in args.Hist else '4g'
        slurm_script = slurm_script.replace('MEM', desired_mem)

        os.makedirs('slurm', exist_ok=True)
        with open("slurm/submit_%s.sh" % uuidstr, 'w') as f:
            f.write(slurm_script)
        print("Submitting job to SLURM...")
        q = subprocess.run(['sbatch', 'slurm/submit_%s.sh' % uuidstr])
        print(q)
        print("Job submitted with UUID:", uuidstr)
        print("Exiting.")
        sys.exit(0)
    elif args.condor:
        with open("templates/condor_exec_template.sh", 'r') as f:
            condor_exec_script = f.read()

        command = command.replace('--condor', '--usexrootd')
        condor_exec_script = condor_exec_script.replace("COMMAND", "python " + command)
        print(condor_exec_script)

        os.makedirs('condor', exist_ok=True)
        with open("condor/exec_%s.sh" % uuidstr, 'w') as f:
            f.write(condor_exec_script)

        with open("templates/condor_sub_template.sh", 'r') as f:
            condor_sub_script = f.read()

        condor_sub_script = condor_sub_script.replace("EXEC_PATH", "condor/exec_%s.sh" % uuidstr)
        condor_sub_script = condor_sub_script.replace("UUID", uuidstr)

        desired_mem = '8GB' if args.Hist == 'transfer' else '4GB'
        desired_cpu = '4' if args.Hist == 'transfer' else '2'
        #desired_mem = "8GB"
        #desired_cpu = "4"
        condor_sub_script = condor_sub_script.replace('REQMEM', desired_mem)
        condor_sub_script = condor_sub_script.replace('REQCPU', desired_cpu)

        with open("condor/submit_%s.sh" % uuidstr, 'w') as f:
            f.write(condor_sub_script)

        print("Submitting job to HTCondor...")
        q = subprocess.run(['condor_submit', 'condor/submit_%s.sh' % uuidstr])
        print(q)
        print("Job submitted with UUID:", uuidstr)
        print("Exiting.")
        sys.exit(0)

from buildEECres4Hists import fill_hist_from_parquet, fill_transferhist_from_parquet, fill_direct_covariance_type1

if args.kinreweight_path is not None:
    from correctionlib import CorrectionSet
    cset = CorrectionSet.from_file(args.kinreweight_path)
    kinreweight_func = cset[args.kinreweight_key].evaluate
else:
    kinreweight_func = None

print("\nBuilding hist...")
if args.Hist.startswith('directcov'):
    H = fill_direct_covariance_type1(thepath, Nboot=0, 
                                     systwt=args.Wtsyst,
                                     nbatch=args.nbatch,
                                     statN=args.statN,
                                     statK=args.statK,
                                     kinreweight=kinreweight_func,
                                     fs=fs)
elif args.Hist == 'transfer':
    print("Clipping bootstrap -> 0")
    H = fill_transferhist_from_parquet(thepath, 
                                       Nboot = 0, 
                                       systwt = args.Wtsyst, 
                                       rng_offset = args.rng,
                                       skipNominal = args.skipNominal,
                                       r123type = args.r123type,
                                       statN = args.statN,
                                       statK = args.statK,
                                       kinreweight = kinreweight_func,
                                       fs = fs)
else:
    H = fill_hist_from_parquet(thepath, args.nboot, args.Wtsyst, 
                               args.rng, args.r123type,
                               prebinned = args.prebinned,
                               nbatch = args.nbatch,
                               skipNominal = args.skipNominal,
                               statN = args.statN,
                               statK = args.statK,
                               kinreweight = kinreweight_func,
                               fs = fs, 
                               collect_debug_info = args.collect_debug_info)
print("Done.\n")


print("Saving result...")
print("Output file: %s" % outfile)
import pickle
with fs.open(outfile, 'wb') as f:
    pickle.dump(H, f)
print("Done.\n")
