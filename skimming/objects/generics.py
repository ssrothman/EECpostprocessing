import awkward as ak

class GenericObjectContainer:
    def __init__(self, events : ak.Array, mandatory_names : dict, optional_names):
        self._events = events
        self._mandatory_nametable = mandatory_names
        self._optional_nametable = optional_names

    #override getattr to provide access to objects by name
    def __getattr__(self, name : str) -> ak.Array | None:
        if name in self._mandatory_nametable:
            objname = self._mandatory_nametable[name]
            result : ak.Array = ak.materialize(self._events[objname])
            return result
        elif name in self._optional_nametable:
            objname = self._optional_nametable[name]
            if hasattr(self._events, objname):
                result : ak.Array = ak.materialize(self._events[objname])
                return result
            else:
                return None
        else:
            raise AttributeError(f"Object {name} not found in GenericObjectContainer")