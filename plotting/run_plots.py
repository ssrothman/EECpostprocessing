import argparse

parser = argparse.ArgumentParser(description='Run plotting scripts.')
parser.add_argument(
    'configs',
    type=str,
    nargs='+',
    help='Path(s) to plotting configuration file(s) (JSON format).'
)
args = parser.parse_args()

import json
from plotting.plotdriver import run_plots

for config_path in args.configs:
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    if 'override_simonplot_config' in cfg:
        import simonplot as splt
        splt.config.update(cfg['override_simonplot_config'])

    for todo in cfg['TO DO']:
        run_plots(todo)