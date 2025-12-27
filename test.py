from coffea.nanoevents import NanoEventsFactory
import json
import awkward as ak

events = NanoEventsFactory.from_root(
    "NANO_selected.root:Events",
    mode='virtual'
).events()

if not isinstance(events, ak.Array):
    raise RuntimeError("Failed to load events as an awkward array!")

with open("skimming/config/objects.json") as f:
    objcfg = json.load(f)

from skimming.objects.objectsetup import AllObjects
allobjs = AllObjects(events, objcfg['objects'], objsyst="")