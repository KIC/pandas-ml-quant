import pandas as pd


class Quant(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def plot(self):
        pass

    @property
    def indicators(self):
        pass

    
