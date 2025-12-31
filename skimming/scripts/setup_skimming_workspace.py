#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="Setup skimming workspace")

parser.add_argument("where", type=str, help="Directory to setup workspace in")
parser.add_argument("runtag", type=str, help="Runtag to process")
parser.add_argument("dataset", type=str, help="Dataset to process")
parser.add_argument("objsyst", type=str, help="Object systematic variation to process")
parser.add_argument("--tables", type=str, nargs='+', required=True,
                    help="Tables to produce in skimming")
parser.add_argument("--output-location", type=str, required=True,
                    help="Location to write output to")
parser.add_argument("--config-suite", type=str, required=True,
                    help="Configuration suite to use for skimming")
args = parser.parse_args()

_ALL_TABLES = [
    "AK4JetKinematicsTable",
    "EventKinematicsTable",
    "ConstituentKinematicsTable",
    "CutflowTable",
    "SimonJetKinematicsTable",
]

if args.tables == ['all']:
    args.tables = _ALL_TABLES

from skimming.config.load_config import load_config
thecfg = load_config(args.config_suite)

from skimming.scaleout.setup_workspace import setup_skim_workspace
setup_skim_workspace(
    working_dir=args.where,
    runtag=args.runtag,
    dataset=args.dataset,
    objsyst=args.objsyst,
    config=thecfg,
    tables=args.tables,
    output_location=args.output_location,
)