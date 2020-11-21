from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant.model.cross_validation.rolling_window_cv import RollingWindowCV


class TestCrossValidation(TestCase):

    def test_rolling_cv(self):
        df = pd.DataFrame({"feature": np.arange(10), "label": np.arange(10)})
        cv = RollingWindowCV(5, 2)

        splits = list(cv.split(df.index))
        # print([s[1] for s in splits])
        print(df)
        print(splits)
        self.assertEqual(3, len(splits))

        # train
        self.assertListEqual([0, 2, 4], [s[0][0] for s in splits])
        self.assertListEqual([4, 6, 8], [s[0][-1] for s in splits])

        # test
        self.assertListEqual([5, 7, 9], [s[1][0] for s in splits])
        self.assertListEqual([6, 8, 9], [s[1][-1] for s in splits])

    def test_rolling5_cv(self):
        df = pd.DataFrame({"feature": np.arange(49), "label": np.arange(49)})
        cv = RollingWindowCV(30, 5)

        splits = list(cv.split(df.index))
        last_train_set = df.index[splits[-1][0]]
        last_test_set = df.index[splits[-1][1]]

        self.assertEqual(30, len(df.index[last_train_set]))
        self.assertLessEqual(len(last_test_set), 4)
        print(last_test_set)

    def test_rolling_cv_empty_last_window(self):
        df = pd.DataFrame({"feature": np.arange(9), "label": np.arange(9)})
        cv = RollingWindowCV(5, 2)

        splits = list(cv.split(df.index))
        # print(df)
        # print(splits)

        self.assertEqual(3, len(splits))

        # train
        self.assertListEqual([0, 2, 4], [s[0][0] for s in splits])
        self.assertListEqual([4, 6, 8], [s[0][-1] for s in splits])

        # test
        self.assertEqual(0, len(splits[-1][1]))

    def test_rooling_cv_forecast_0(self):
        df = pd.DataFrame({"feature": np.arange(4), "label": np.arange(4)})
        cv = RollingWindowCV(window=1, retrain_after=0)

        splits = list(cv.split(df.index))
        #print(df)
        #print(splits)

        self.assertEqual(4, len(splits))

        # train
        self.assertListEqual([0, 1, 2, 3], [s[0][0] for s in splits])
        self.assertEqual(df.index[-1], df.index[splits[-1][0][-1]])

        # test
        self.assertListEqual([1, 2, 3], [s[1][-1] for s in splits if len(s[1]) > 0])
        self.assertEqual(0, len(splits[-1][1]))