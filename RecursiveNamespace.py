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

    def to_dict(self):
        ans = {}
        for key, val in self.__dict__.items():
            if isinstance(val, RecursiveNamespace):
                ans[key] = val.to_dict()
            else:
                ans[key] = val
        return ans

    def __add__(self, other):
        return self

    def update(self, newdict):
        for key, val in newdict.items():
            if hasattr(self, key):
                if isinstance(val, dict):
                    if isinstance(getattr(self, key), RecursiveNamespace):
                        getattr(self, key).update(val)
                    else:
                        raise ValueError("Can't update a leaf with a dict")
                elif isinstance(val, list):
                    getattr(self, key).extend(val)
                elif isinstance(val, tuple):
                    setattr(self, key, (*getattr(self, key), *val))
                elif isinstance(val, set):
                    getattr(self, key).update(val)
                else:
                    print("WARNING: OVERWRITING %s"%key)
                    setattr(self, key, val)
            else:
                if isinstance(val, dict):
                    setattr(self, key, RecursiveNamespace(**val))
                else:
                    setattr(self, key, val)
