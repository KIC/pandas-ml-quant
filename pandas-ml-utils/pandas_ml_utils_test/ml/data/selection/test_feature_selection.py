from unittest import TestCase

from pandas_ml_utils.ml.data.selection.feature_selection import score_feature
from test.config import DF_TEST


class TestFeatureSelection(TestCase):

    def test_continuous_feature_continuous_label(self):
        score = score_feature(DF_TEST, "Close", "Open", [1, 2, 3])
        print(score.scores)

    def test_discrete_feature_continuous_label(self):
        df = DF_TEST.copy()
        df["discrete"] = (df["Open"] / df["Close"] * 10).round().astype(int)
        score = score_feature(df, "Close", "discrete", [1, 2, 3])
        print(score.scores)

    def test_continuous_feature_discrete_label(self):
        df = DF_TEST.copy()
        df["discrete"] = (df["Open"] / df["Close"] * 10).round().astype(int)
        score = score_feature(df, "discrete", "Close", [1, 2, 3])
        print(score.scores)

    def test_discrete_feature_discrete_label(self):
        df = DF_TEST.copy()
        df["discrete"] = (df["Open"] / df["Close"] * 10).round().astype(int)
        score = score_feature(df, "discrete", "discrete", [1, 2, 3])
        print(score.scores)
