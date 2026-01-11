import json
import os
from general.fslookup.files import get_rootfiles
from general.fslookup.location_lookup import location_lookup

with open(os.path.join(os.path.dirname(__file__), 'datasets.json')) as f:
    cfg = json.load(f)

def lookup_dataset(runtag : str, dataset : str) -> dict:
    return cfg[runtag][dataset]

def get_JERC_era(runtag : str, dataset : str) -> str:
    dsetcfg = cfg[runtag][dataset]
    return dsetcfg['era']

def get_flags(runtag : str, dataset : str) -> dict:
    dsetcfg = cfg[runtag][dataset]
    return dsetcfg['flags']

def get_target_files(runtag : str, dataset : str, exclude_dropped=True):
    base = cfg[runtag]['base']

    dsetcfg = cfg[runtag][dataset]

    tag = dsetcfg['tag']
    location = dsetcfg['location']

    fs, rootpath = location_lookup(location)
    if type(tag) not in [list, tuple]:
        tag = [tag]

    allfiles = []

    for t in tag:
        root = os.path.join(rootpath, base, t)
        allfiles += get_rootfiles(
            fs, root, 
            exclude_dropped=exclude_dropped
        )
    
    return allfiles, location
