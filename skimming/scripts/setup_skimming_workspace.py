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
parser.add_argument("--nocheck", action='store_true', help="Skip checking for existing outputs before setting up workspace")
args = parser.parse_args()

_ALL_KINEMATICS_TABLES = [
    "AK4JetKinematicsTable",
    "EventKinematicsTable",
    "ConstituentKinematicsTable",
    "CutflowTable",
    "SimonJetKinematicsTable",
]
_ALL_RES4TEE_TABLES = [
    "EECres4Obs:True,tee,total",
    "EECres4Obs:True,tee,unmatched",
    "EECres4Obs:True,tee,untransfered",
    "EECres4Obs:False,tee,total",
    "EECres4Obs:False,tee,unmatched",
    "EECres4Obs:False,tee,untransfered",
    "EECres4Transfer:tee"
]
_ALL_RES4DIPOLE_TABLES = [
    "EECres4Obs:True,dipole,total",
    "EECres4Obs:True,dipole,unmatched",
    "EECres4Obs:True,dipole,untransfered",
    "EECres4Obs:False,dipole,total",
    "EECres4Obs:False,dipole,unmatched",
    "EECres4Obs:False,dipole,untransfered",
    "EECres4Transfer:dipole"
]
_ALL_RES4TRIANGLE_TABLES = [
    "EECres4Obs:True,triangle,total",
    "EECres4Obs:True,triangle,unmatched",
    "EECres4Obs:True,triangle,untransfered",
    "EECres4Obs:False,triangle,total",
    "EECres4Obs:False,triangle,unmatched",
    "EECres4Obs:False,triangle,untransfered",
    "EECres4Transfer:triangle"
]
_ALL_RES4_TABLES = _ALL_RES4TEE_TABLES + _ALL_RES4DIPOLE_TABLES + _ALL_RES4TRIANGLE_TABLES

if args.tables == ['allKinematics']:
    args.tables = _ALL_KINEMATICS_TABLES
elif args.tables == ['allRes4']:
    args.tables = _ALL_RES4_TABLES
elif args.tables == ['allRes4tee']:
    args.tables = _ALL_RES4TEE_TABLES
elif args.tables == ['allRes4dipole']:
    args.tables = _ALL_RES4DIPOLE_TABLES
elif args.tables == ['allRes4triangle']:
    args.tables = _ALL_RES4TRIANGLE_TABLES

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
    nocheck=args.nocheck
)