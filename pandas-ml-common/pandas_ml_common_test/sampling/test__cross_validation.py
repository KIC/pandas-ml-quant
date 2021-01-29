from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pandas_ml_common.sampling.cross_validation import KEquallyWeightEvents, KFoldBoostRareEvents, \
    PartitionedOnRowMultiIndexCV


class TestCrossValidation(TestCase):

    def test__boost_rare_event(self):
        x = np.array([0, 0, 0, 1])
        cv = KFoldBoostRareEvents(n_splits=2)
        indices = list(cv.split(x, x))

        self.assertEqual(len(indices), 2)
        self.assertGreaterEqual(x[indices[0][1]].sum(), 1)
        self.assertGreaterEqual(x[indices[1][1]].sum(), 1)

    def test__equal_weight_events(self):
        x = pd.DataFrame({"a": np.array([0, 0, 0, 1])})
        cv = KEquallyWeightEvents(n_splits=2)
        indices = list(cv.split(x, x))

        self.assertEqual(len(indices), 2)
        self.assertGreaterEqual(x.loc[indices[0][1]].values.sum(), 1)
        self.assertGreaterEqual(x.loc[indices[1][1]].values.sum(), 1)

    def test__partitioning_by_multi_index_row(self):
        df = pd.DataFrame({"a": [1, 2] * 2})
        df.index = pd.MultiIndex.from_product([["A", "B"], [1, 2]])

        cv = PartitionedOnRowMultiIndexCV(KFold(2))
        print(list(KFold(2).split(df.loc["A"])))

        indices = list(cv.split(df.index, y=df))
        fold_A = indices[:1]
        fold_B = indices[2:]

        self.assertEqual(len(indices), len(list(KFold(2).split(df.loc["A"])))*2)

        for idxs in fold_A:
            for idx in idxs:
                self.assertIn("A", df.iloc[idx].index)
                self.assertNotIn("B", df.iloc[idx].index)

        for idxs in fold_B:
            for idx in idxs:
                self.assertIn("B", df.iloc[idx].index)
                self.assertNotIn("A", df.iloc[idx].index)
