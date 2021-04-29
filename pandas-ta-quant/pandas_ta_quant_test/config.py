import os

import numpy as np

from pandas_ta_quant import pd

print('NUMPY VERSION', np.__version__)


DF_TEST_MULTI = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "spy_gld.pickle"))
DF_TEST = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "SPY.csv"), index_col='Date', parse_dates=True)
DF_TEST2 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "GLD.csv"), index_col='Date', parse_dates=True)
DF_TEST_MULTI_ROW = pd.concat([DF_TEST.add_multi_index("SPY", axis=0), DF_TEST2.add_multi_index("GLD", axis=0)], axis=0)
DF_TEST_MULTI_ROW_MULTI_COLUMN = pd.concat([DF_TEST_MULTI.add_multi_index("A", axis=0), DF_TEST_MULTI.add_multi_index("B", axis=0)], axis=0)
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})

DF_INVERSE_GAF = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "inverse_gaf.df"))
DF_TEST_MULTI_CLASS = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "one_hot_classified_df.pickle"))

CSV_TRADES = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "trades.csv")

