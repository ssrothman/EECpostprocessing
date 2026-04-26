import json
import fsspec
import os
import hashlib
import socket
import time

# Load the location lookup map
# wrt this file
with open(os.path.join(os.path.dirname(__file__), 'location_lookup.json')) as f:
    lookupmap = json.load(f)

if 'xrootd-submit' in lookupmap:
    # if we have an xrootd-submit entry, add a hack 
    # to randomly assign to one of the submit gateways
    # the hostid should be submit%d.mit.edu 
    # where %d is randomly between 50 and 59 (inclusive)

    lookupmap['xrootd-submit'][0] = f"submit{50 + (time.time_ns() % 10)}.mit.edu"
    print("[Info] Updated xrootd-submit hostid to:", lookupmap['xrootd-submit'][0])

def location_lookup(location : str):
    hostid, rootpath = lookupmap[location]

    if hostid == 'local':
        fs = fsspec.filesystem('file')
    else:
        from fsspec_xrootd import XRootDFileSystem
        fs = XRootDFileSystem(hostid, timeout=300) # long timeout to avoid issues with large / slow transfers. does this work ? lol
    
    return fs, rootpath

def lookup_hostid(location : str):
    hostid, _ = lookupmap[location]
    return hostid