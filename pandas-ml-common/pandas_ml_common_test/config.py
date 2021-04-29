import os

import numpy as np

from pandas_ml_common import pd

print('NUMPY VERSION', np.__version__)


def _with_multi_index(df, header):
    df = df.copy()
    df.columns = pd.MultiIndex.from_product([[header], df.columns])
    return df


def _with_multi_index_row(df, header):
    df = df.copy()
    df.index = pd.MultiIndex.from_product([[header], df.index])
    return df


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
TEST_FILE = os.path.join(DATA_PATH, "SPY.csv")
TEST_DF = pd.read_csv(TEST_FILE, index_col='Date', parse_dates=True)
TEST_MULTI_INDEX_DF = _with_multi_index(TEST_DF, "A").join(_with_multi_index(TEST_DF, "B"))
TEST_MUTLI_INDEX_ROW_DF = pd.concat([_with_multi_index_row(TEST_DF.tail(), "A"), _with_multi_index_row(TEST_DF.tail() + 1.0, "B")], axis=0)


