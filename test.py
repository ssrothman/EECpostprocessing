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

from skimming.objects.AllObjects import AllObjects
allobjs = AllObjects(
    events,
    "MC",
    objcfg['objects'], 
    btagcfg['btagging'],
    JECcfg['JERC'],
    objsyst="nominal"
)