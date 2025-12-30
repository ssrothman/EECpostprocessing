import json
import os
from skimming.files import get_rootfiles

with open("skimming/datasets/datasets.json") as f:
    cfg = json.load(f)

with open("skimming/datasets/location_lookup.json") as f:
    location_lookup = json.load(f)

def get_target_files(runtag : str, dataset : str, exclude_dropped=True):
    base = cfg[runtag]['base']

    dsetcfg = cfg[runtag][dataset]

    tag = dsetcfg['tag']
    location = dsetcfg['location']

    hostid, rootpath = location_lookup[location]
    if type(tag) not in [list, tuple]:
        tag = [tag]

    allfiles = []

    for t in tag:
        root = os.path.join(rootpath, base, t)
        allfiles += get_rootfiles(
            hostid, root, 
            exclude_dropped=exclude_dropped
        )
    
    return allfiles
