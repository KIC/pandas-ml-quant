import os

import numpy as np

from pandas_ml_common import pd


DF_NOTES = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "banknote_authentication.csv"))
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})
