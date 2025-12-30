from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import json
import awkward as ak

NanoAODSchema.warn_missing_crossrefs = False

events = NanoEventsFactory.from_root(
    "NANO_selected.root:Events",
    mode='virtual'
).events()

if not isinstance(events, ak.Array):
    raise RuntimeError("Failed to load events as an awkward array!")

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

from skimming.skim import skim
skim(
    events,
    thecfg,
    "test_output",
    ['count']
)
skim(
    events,
    thecfg,
    "test_output",
    [
        'AK4JetKinematicsTable',
        'ConstituentKinematicsTable',
        'CutflowTable',
        'EventKinematicsTable',
        'SimonJetKinematicsTable'
    ]
)