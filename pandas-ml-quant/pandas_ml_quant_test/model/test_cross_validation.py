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

        self.assertEqual(2, len(splits))
        self.assertListEqual([0, 2], [s[0][0] for s in splits])
        self.assertListEqual([4, 6], [s[0][-1] for s in splits])
        self.assertListEqual([5, 7], [s[1][0] for s in splits])
        self.assertListEqual([6, 8], [s[1][-1] for s in splits])
