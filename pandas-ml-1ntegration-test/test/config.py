import os

import numpy as np

from pandas_ml_common import pd

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
DF_TEST = pd.read_csv(os.path.join(TEST_DATA_PATH, "SPY.csv"), index_col='Date')
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})
