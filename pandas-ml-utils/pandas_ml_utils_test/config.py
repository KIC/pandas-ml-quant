import os

import numpy as np

from pandas_ml_common import pd

TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
DF_NOTES = pd.read_csv(os.path.join(TEST_DATA_PATH, "banknote_authentication.csv"))
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})
DF_TEST = pd.read_csv(os.path.join(TEST_DATA_PATH, "SPY.csv"), index_col='Date')
DF_SUMMARY = pd.read_pickle(os.path.join(TEST_DATA_PATH, "multi_index_row_summary.df"))