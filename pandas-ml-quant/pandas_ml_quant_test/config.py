import os

import numpy as np

from pandas_ml_common import pd


DF_TEST_MULTI = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "spy_gld.pickle"))
DF_TEST = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "SPY.csv"), index_col='Date', parse_dates=True)
DF_DEBUG = pd.DataFrame({"Close": np.random.random(10)})

DF_INVERSE_GAF = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "inverse_gaf.df"))
DF_TEST_MULTI_CASS = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "one_hot_classified_df.pickle"))