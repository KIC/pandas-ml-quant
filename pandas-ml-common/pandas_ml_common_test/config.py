import os
from pandas_ml_common import pd


def _with_multi_index(df, header):
    df = df.copy()
    df.columns = pd.MultiIndex.from_product([[header], df.columns])
    return df


TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "SPY.csv")
TEST_DF = pd.read_csv(TEST_FILE, index_col='Date', parse_dates=True)
TEST_MULTI_INDEX_DF = _with_multi_index(TEST_DF, "A").join(_with_multi_index(TEST_DF, "B"))


