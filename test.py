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
from skimming.weights.StandardWeights import StandardWeights

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
stdwts = StandardWeights(weightscfg['eventweight']['params'], evtselcfg['eventsel']['params'])
weights = stdwts.get_weights(allobjs)