import json
import fsspec
import os

# Load the location lookup map
# wrt this file
with open(os.path.join(os.path.dirname(__file__), 'location_lookup.json')) as f:
    lookupmap = json.load(f)

def location_lookup(location : str):
    hostid, rootpath = lookupmap[location]

    if hostid == 'local':
        fs = fsspec.filesystem('file')
    else:
        from fsspec_xrootd import XRootDFileSystem
        fs = XRootDFileSystem(hostid, timeout=120)
    
    return fs, rootpath

def lookup_hostid(location : str):
    hostid, _ = lookupmap[location]
    return hostid