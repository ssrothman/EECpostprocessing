import json
import os
from skimming.config.load_config import load_config

thecfg = load_config('basic')

from skimming.scaleout.setup_workspace import setup_skim_workspace
setup_skim_workspace(
    working_dir="test_skimscript",
    runtag="Apr_23_2025",
    dataset="Pythia_inclusive",
    objsyst='nominal',
    config=thecfg,
    tables=["AK4JetKinematicsTable", 
            "EventKinematicsTable",
            "ConstituentKinematicsTable",
            "CutflowTable",
            "SimonJetKinematicsTable"],
    output_location="local-submit",
)
setup_skim_workspace(
    working_dir="test_skimscript2",
    runtag="Apr_23_2025",
    dataset="Pythia_inclusive",
    objsyst='nominal',
    config=thecfg,
    tables=["count"],
    output_location="local-submit",
)