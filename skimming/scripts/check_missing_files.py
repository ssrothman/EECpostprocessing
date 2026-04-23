#!/usr/bin/env python

import argparse
import os.path
import json
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from general.datasets.datasets import get_target_files
from general.fslookup.location_lookup import location_lookup, lookup_hostid
from skimming.objects.AllObjects import AllObjects

# Central cache configuration
CACHE_DIR = os.path.expanduser("~/.cache/eec_skimming")
CACHE_FILE = os.path.join(CACHE_DIR, "uniqueid_cache.json")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_cache():
    ensure_cache_dir()
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_cache(cache):
    ensure_cache_dir()
    # Use fcntl lock for process-safe writes
    with open(CACHE_FILE, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(cache, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

parser = argparse.ArgumentParser(description="Use skimming results to determine which input root files remain to be processed")
parser.add_argument('Runtag', type=str, help="Runtag of the skimming workspace to check")
parser.add_argument('Dataset', type=str, help="Dataset name of the skimming workspace to check")
parser.add_argument('Objsyst', type=str, help="Objsyst of the skimming workspace to check")
parser.add_argument('Location', type=str, help="Location of the skimming workspace to check")
parser.add_argument('ConfigSuite', type=str, help="Configsuite of the skimming workspace to check")
parser.add_argument('table', type=str, help="Name of the skim table to check")
parser.add_argument('-j', type=int, default=1, help="Number of parallel workers for input-file checks (default: 1)")
parser.add_argument('--write-missing', type=str, default=None, help="Write missing input files to this file")
parser.add_argument('--write-glitched', type=str, default=None, help="Write glitched skim results (unmatched files) to this file")
args = parser.parse_args()

target_files, dset_location = get_target_files(args.Runtag, args.Dataset, exclude_dropped=args.table != 'count')
hostid = lookup_hostid(dset_location)

skimfs, skimbase = location_lookup(args.Location)
skimpath = os.path.join(
    skimbase, 
    args.ConfigSuite,
    args.Runtag,
    args.Dataset,
    args.Objsyst,
    args.table
)

listdir = skimfs.listdir(skimpath)

skimresults = set()
for item in listdir:
    if not (item['name'].endswith('.parquet') or item['name'].endswith('.json')):
        continue
    if item['name'] == 'merged.json':
        continue
    if item['size'] == 0:
        print("WARNING: Found empty skim result file %s, skipping." % item['name'])
        continue
    skimresults.add(os.path.basename(item['name']))

print("Target files:", len(target_files))
print("Skim files:", len(skimresults))

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

# Load cache before processing
uniqueid_cache = load_cache()
print("Uniqueid cache loaded: %d entries from %s" % (len(uniqueid_cache), CACHE_FILE))

missing_files = set()

def get_expected_name(target_file, hostid, uniqueid_cache):
    # Check cache first
    if target_file in uniqueid_cache:
        cached_entry = uniqueid_cache[target_file]
        if 'uniqueid' in cached_entry:
            # Use cached value
            return target_file, cached_entry['uniqueid'] + ".parquet", None
    
    # Cache miss: read ROOT file and extract uniqueid
    events = NanoEventsFactory.from_root('root://' + hostid + '//' + target_file + ":Events", schemaclass=NanoAODSchema).events()
    uniqueid = AllObjects.get_uniqueid(events) # type: ignore
    # Return the computed uniqueid for cache update
    return target_file, uniqueid + ".parquet", uniqueid

max_workers = max(1, args.j)
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    future_to_target_file = {executor.submit(get_expected_name, target_file, hostid, uniqueid_cache): target_file for target_file in target_files}
    for future in tqdm(as_completed(future_to_target_file), total=len(target_files)):
        target_file, expected_name, computed_uniqueid = future.result()
        
        # Update cache if we computed a new uniqueid
        if computed_uniqueid is not None:
            uniqueid_cache[target_file] = {'uniqueid': computed_uniqueid}
        
        if expected_name in skimresults:
            # success!
            skimresults.remove(expected_name) # remove from the set of skim results so that 
                                              # at the end, any remaining skim results in the set 
                                              # are unexpected/skimming errors
        else:
            missing_files.add(target_file)

# Save updated cache with lock
save_cache(uniqueid_cache)
print("Uniqueid cache saved with %d entries" % len(uniqueid_cache))

print("Missing files:", len(missing_files))
for f in sorted(missing_files):
    print(f)

if args.write_missing and len(missing_files) > 0:
    with open(args.write_missing, 'w') as f:
        for target_file in sorted(missing_files):
            f.write(target_file + '\n')
    print("Missing files written to %s" % args.write_missing)

if len(skimresults) > 0:
    print("WARNING: Found %d skim result files that do not correspond to any target file:" % len(skimresults))
    for s in sorted(skimresults):
        print("  ", s)
    
    if args.write_glitched:
        with open(args.write_glitched, 'w') as f:
            for skim_file in sorted(skimresults):
                f.write(skim_file + '\n')
        print("Glitched skim results written to %s" % args.write_glitched)