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

from skimming.objects import jets
from skimming.objects.AllObjects import AllObjects
allobjs = AllObjects(
    events,
    "MC",
    objcfg['objects'], 
    btagcfg['btagging'],
    JECcfg['JERC'],
    objsyst="nominal"
)

from skimming.selections.factories import runEventSelection, runJetSelection
from skimming.weights.factory import runWeightsFactory

eventselection = runEventSelection(
    evtselcfg['eventsel'],
    allobjs,
    flags={}
)
jetselection = runJetSelection(
    jetselcfg['jetsel'],   
    allobjs,
    eventselection, 
    flags={}
)
weights = runWeightsFactory(weightscfg['eventweight'], evtselcfg['eventsel'], allobjs)


from skimming.tables.driver import TableDriver
driver = TableDriver(
    [
        'AK4JetKinematicsTable',
        'ConstituentKinematicsTable',
        'CutflowTable',
        'EventKinematicsTable',
        'SimonJetKinematicsTable'
    ],
    'test_output'
)
driver.run_tables(
    allobjs,
    eventselection,
    jetselection,
    weights
)