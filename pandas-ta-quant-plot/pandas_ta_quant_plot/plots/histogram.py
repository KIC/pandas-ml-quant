import pandas as pd
import numpy as np

class Hist(object):

    def __init__(self, bins: int = 20):
        self.buckets = bins

