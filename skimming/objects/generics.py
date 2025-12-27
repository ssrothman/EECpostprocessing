import awkward as ak

class GenericObjectContainer:
    def __init__(self, events : ak.Array, nametable : dict):
        self._events = events
        self._nametable = nametable

    #override getattr to provide access to objects by name
    def __getattr__(self, name : str) -> ak.Array:
        if name in self._nametable:
            objname = self._nametable[name]
            result : ak.Array = ak.materialize(self._events[objname])
            return result
        else:
            raise AttributeError(f"Object {name} not found in GenericObjectContainer")