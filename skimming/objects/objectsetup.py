import awkward as ak
from vector import obj
from .jets import Jets
from .muons import Muons
from .generics import GenericObjectContainer

from typing import Any

objclasses = {
    "Jets" : Jets,
    "Muons" : Muons,
    "GenericObjectContainer" : GenericObjectContainer
}

class AllObjects:
    def __init__(self, 
                events : ak.Array, 
                cfg : dict,
                objsyst : str):
        
        self._objects = {}
        for objname, objcfg in cfg.items():
            clsname = objcfg['class']
            cls = objclasses[clsname]

            self._objects[objname] = cls(
                events,
                **objcfg['params']
            )

        self._objects['isMC'] = hasattr(events, 'Generator')

    def __getattr__(self, name : str) -> Any:
        if name in self._objects:
            return self._objects[name]
        else:
            raise AttributeError("Object %s not loaded!"%name)