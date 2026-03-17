import sys
import os
import fsspec

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from skimming.skim import skim
from skimming.config.load_config import load_config

NanoAODSchema.warn_missing_crossrefs = False

config = load_config('evtMC_proj')

config['era']      = 'skip'
config['objsyst']  = 'NOM'
config['flags']    = {}
config['btagging'] = {}
config['JERC']     = {}

tables = [
    'EECprojObs:False,total',
    'EECprojObs:True,total',
    'EECprojObs:False,unmatched',
    'EECprojObs:True,unmatched',
    'EECprojTransfer',
]

output_dir = os.path.join(os.path.dirname(__file__), 'skim_output')
fs = fsspec.filesystem('file')
fs.makedirs(output_dir, exist_ok=True)

files = sys.argv[1:] if len(sys.argv) > 1 else ['NANO_selected.root', 'NANO_dropped.root']

for fpath in files:
    if not os.path.exists(fpath):
        print(f'Skipping {fpath} (not found)')
        continue

    print(f'\n=== Skimming {fpath} ===')
    events = NanoEventsFactory.from_root(
        fpath + ':Events',
        schemaclass=NanoAODSchema
    ).events()

    if len(events) == 0:
        print('  No events, skipping.')
        continue

    skim(events, config, output_dir, fs, tables)

print('\nDone. Output written to', output_dir)
