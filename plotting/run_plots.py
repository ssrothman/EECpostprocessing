import argparse

parser = argparse.ArgumentParser(description='Run plotting scripts.')
parser.add_argument('config', type=str, help='Path to the plotting configuration file (JSON format).')
args = parser.parse_args()

import json
from plotting.plotdriver import run_plots

with open(args.config, 'r') as f:
    cfg = json.load(f)

for todo in cfg['TO DO']:
    run_plots(todo)