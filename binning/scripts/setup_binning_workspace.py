#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser(description="Setup binning workspace for scaleout execution")
parser.add_argument("where", type=str, help="Directory where workspace will be created")
parser.add_argument("runtag", type=str, help="Runtag to process")

parser.add_argument("--mc", type=str, nargs="*", default=[], help="MC datasets (expanded over --objsysts and --wtsysts)")
parser.add_argument("--data", type=str, nargs="*", default=[], help="Data datasets (uses objsyst=DATA and wtsyst=nominal)")

parser.add_argument("--objsysts", type=str, nargs="+", default=["nominal"], help="Object systematics to process")
parser.add_argument("--wtsysts", type=str, nargs="+", default=["nominal"], help="Weight systematics to process")

parser.add_argument("--tables", type=str, nargs="+", required=True, help="Table names to process")

parser.add_argument("--location", type=str, default="local-submit", help="Output location")
parser.add_argument("--config-suite", type=str, default="BasicConfig", help="Configuration suite")

parser.add_argument("--statN", type=int, default=-1, help="N for stat split (-1 to disable)")
parser.add_argument("--statK", type=int, default=-1, help="K for stat split")

parser.add_argument("--bincfg", type=str, default=None, help="Optional explicit binning config name")

parser.add_argument("--cov", action="store_true", help="Build covariance output")

parser.add_argument("--nocheck", action="store_true", help="Skip existing-output checks")

args = parser.parse_args()

_ALL_HT_DATASETS = [
    'Pythia_HT-0to70',
    'Pythia_HT-70to100',
    'Pythia_HT-100to200',
    'Pythia_HT-200to400',
    'Pythia_HT-400to600',
    'Pythia_HT-600to800',
    'Pythia_HT-800to1200',
    'Pythia_HT-1200to2500',
    'Pythia_HT-2500toInf',
]
if args.mc == ['allHT']:
    args.mc = _ALL_HT_DATASETS

from skimming.tables.expand_tables import expand_tables, table_names
args.tables = table_names(expand_tables(args.tables))

from binning.scaleout.setup_workspace import setup_binning_workspace

dataset_objsyst_wtsyst_triples: list[tuple[str, str, str]] = []

for dataset in args.mc:
    for objsyst in args.objsysts:
        for wtsyst in args.wtsysts:
            dataset_objsyst_wtsyst_triples.append((dataset, objsyst, wtsyst))

for dataset in args.data:
    dataset_objsyst_wtsyst_triples.append((dataset, "DATA", "nominal"))

if len(dataset_objsyst_wtsyst_triples) == 0:
    parser.error("No datasets specified. Use one or both of --mc/--data")

# remove duplicates
dataset_objsyst_wtsyst_triples = list(set(dataset_objsyst_wtsyst_triples))

ncommands = setup_binning_workspace(
    working_dir=args.where,
    runtag=args.runtag,
    tables=args.tables,
    location=args.location,
    config_suite=args.config_suite,
    statN=args.statN,
    statK=args.statK,
    bincfg=args.bincfg,
    cov=args.cov,
    nocheck=args.nocheck,
    dataset_objsyst_wtsyst_triples=dataset_objsyst_wtsyst_triples,
)

print(f"Workspace written to {args.where}")
print(f"Number of commands: {ncommands}")
