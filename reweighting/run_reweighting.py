from argparse import ArgumentParser


parser = ArgumentParser(description='Run reweighting')
parser.add_argument('specs', nargs='+', help='Reweighting spec files to run')

args = parser.parse_args()

import json
from reweighting.driver import run_reweighting
from reweighting.spec import reweighting_spec

for spec_file in args.specs:
    with open(spec_file, 'r') as f:
        spec : reweighting_spec = json.load(f)

    run_reweighting(spec)