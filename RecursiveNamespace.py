import json

class RecursiveNamespace:
    @staticmethod 
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val,list) or isinstance(val,set) or isinstance(val,tuple):
                setattr(self, key, [RecursiveNamespace.map_entry(entry) for entry in val])
            else:
                setattr(self, key, val)

    def __add__(self, other):
        return self
