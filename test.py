import json

with open("skimming/config/objects.json") as f:
    objcfg = json.load(f)
with open("skimming/config/btag.json") as f:
    btagcfg = json.load(f)
with open("skimming/config/JERC.json") as f:
    JECcfg = json.load(f)
with open("skimming/config/eventsel.json") as f:
    evtselcfg = json.load(f)
with open("skimming/config/jetsel.json") as f:
    jetselcfg = json.load(f)
with open("skimming/config/weights.json") as f:
    weightscfg = json.load(f)

thecfg = objcfg
thecfg.update(btagcfg)
thecfg.update(JECcfg)
thecfg.update(evtselcfg)
thecfg.update(jetselcfg)
thecfg.update(weightscfg)

from skimming.scaleout.make_skimscript import make_skimscript
make_skimscript(
    working_dir="test_skimscript",
    runtag="Apr_23_2025",
    dataset="Pythia_inclusive",
    config=thecfg,
    tables=["AK4JetKinematicsTable", 
            "EventKinematicsTable",
            "ConstituentKinematicsTable",
            "CutflowTable",
            "SimonJetKinematicsTable"],
    output_location="local-submit",
)
make_skimscript(
    working_dir="test_skimscript2",
    runtag="Apr_23_2025",
    dataset="Pythia_inclusive",
    config=thecfg,
    tables=["count"],
    output_location="local-submit",
)