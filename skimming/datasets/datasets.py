import json
import os
from skimming.files import get_rootfiles

with open("skimming/datasets/datasets.json") as f:
    cfg = json.load(f)

location_lookup = {
    'simon-LPC' : (
        'cmseos.fnal.gov',
        '/store/group/lpcpfnano/srothman/'
    )
}

def get_target_files(runtag : str, dataset : str, exclude_dropped=True):
    dsetcfg = cfg[runtag][dataset]

    tag = dsetcfg['tag']
    location = dsetcfg['location']
    #era = dsetcfg['era']
    #flags = dsetcfg['flags']

    hostid, rootpath = location_lookup[location]
    if type(tag) not in [list, tuple]:
        tag = [tag]

    allfiles = []

    for t in tag:
        rootpath = os.path.join(rootpath, t)
        allfiles += get_rootfiles(
            hostid, rootpath, 
            exclude_dropped=exclude_dropped
        )
    
    return allfiles
