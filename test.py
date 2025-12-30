import json
import os

base = os.path.dirname(__file__)+'/'

with open(base+"skimming/config/objects.json") as f:
    objcfg = json.load(f)
with open(base+"skimming/config/btag.json") as f:
    btagcfg = json.load(f)
with open(base+"skimming/config/JERC.json") as f:
    JECcfg = json.load(f)
with open(base+"skimming/config/eventsel.json") as f:
    evtselcfg = json.load(f)
with open(base+"skimming/config/jetsel.json") as f:
    jetselcfg = json.load(f)
with open(base+"skimming/config/weights.json") as f:
    weightscfg = json.load(f)

thecfg = objcfg
thecfg.update(btagcfg)
thecfg.update(JECcfg)
thecfg.update(evtselcfg)
thecfg.update(jetselcfg)
thecfg.update(weightscfg)

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