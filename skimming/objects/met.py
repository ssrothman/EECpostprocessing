from typing import Any, List
import awkward as ak

class MET:
    def __init__(self,
                  events : ak.Array,
                    name : str,
                      paramsuffix : str = ""):
        
        self._obj = events[name]
        self._paramsuffix = paramsuffix

    def __getattr__(self, name: str) -> Any:
        if name + self._paramsuffix not in self._obj.fields:
            raise AttributeError(f"'MET' object has no attribute '{name}'")
        return ak.materialize(self._obj[name + self._paramsuffix])
    
