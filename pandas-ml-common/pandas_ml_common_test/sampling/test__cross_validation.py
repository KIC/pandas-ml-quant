from unittest import TestCase

import numpy as np

from pandas_ml_common.sampling.cross_validation import KEquallyWeightEvents, KFoldBoostRareEvents


class TestCrossValidation(TestCase):

    def test__boost_rare_event(self):
        x = np.array([0, 0, 0, 1])
        cv = KFoldBoostRareEvents(n_splits=2)
        indices = list(cv.split(x, x))

        self.assertEqual(len(indices), 2)
        self.assertGreaterEqual(x[indices[0][1]].sum(), 1)
        self.assertGreaterEqual(x[indices[1][1]].sum(), 1)

    def test__equal_weight_events(self):
        x = np.array([0, 0, 0, 1])
        cv = KEquallyWeightEvents(n_splits=2)
        indices = list(cv.split(x, x))

        self.assertEqual(len(indices), 2)
        self.assertGreaterEqual(x[indices[0][1]].sum(), 1)
        self.assertGreaterEqual(x[indices[1][1]].sum(), 1)

