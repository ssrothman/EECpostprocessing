import argparse
from coffea.nanoevents import NanoEventsFactory

import json
with open("filelist.json", 'r') as f:
    filelist = json.load(f)

import sys
N = int(sys.argv[1])
files = filelist[str(N)]

with open("args.json", 'r') as f:
    EECargs = json.load(f)

from RecursiveNamespace import RecursiveNamespace
EECargs['config'] = RecursiveNamespace(**EECargs['config'])

print("Read args")

from iadd import iadd

result = None
for file in files:
    print(file)
    events = NanoEventsFactory.from_root(file).events()
    print("got events")

    from processing.EECProcessor import EECProcessor
    processor = EECProcessor(**EECargs)
    print("made processor")
    if result is None:
        result = processor.process(events)
    else:
        iadd(result, processor.process(events))
    print("processed")

with open("result.pkl", 'wb') as f:
    import pickle
    pickle.dump(result, f)

print("done")
