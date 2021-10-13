import os

import numpy as np

from pandas_ml_common import pd

print('NUMPY VERSION', np.__version__)


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
TEST_FILE = os.path.join(DATA_PATH, "SPY.csv")
TEST_DF = pd.read_csv(TEST_FILE, index_col='Date', parse_dates=True)


