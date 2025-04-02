import pandas as pd

class DataFrameAccumulator:
    def __init__(self, valdict=None):
        self.df = pd.DataFrame(valdict)

    def __add__(self, other):
        self.df = pd.concat([self.df, other.df], ignore_index=True)
        return self
