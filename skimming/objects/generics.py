import awkward as ak
from typing import Optional

class GenericObjectContainer:
    def __init__(self, events : ak.Array, mandatory_names : dict, optional_names):
        self._events = events
        self._mandatory_nametable = mandatory_names
        self._optional_nametable = optional_names

    def _navigate(self, path : str) -> Optional[ak.Array]:
        """Navigate a dot-separated field path in events (e.g. 'LHE.HT')."""
        parts = path.split('.')
        obj = self._events
        for part in parts:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return ak.materialize(obj)

    #override getattr to provide access to objects by name
    def __getattr__(self, name : str) -> Optional[ak.Array]:
        if name in self._mandatory_nametable:
            objname = self._mandatory_nametable[name]
            result : ak.Array = ak.materialize(self._events[objname])
            return result
        elif name in self._optional_nametable:
            objname = self._optional_nametable[name]
            result = self._navigate(objname)
            return result
        else:
            raise AttributeError(f"Object {name} not found in GenericObjectContainer")
