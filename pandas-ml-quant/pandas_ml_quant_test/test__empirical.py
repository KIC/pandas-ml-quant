from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_ml_quant.empirical import ECDF


class TestEmpirical(TestCase):

    def test_empirical_probs(self):
        ecdf = ECDF([1, 2, 2, 3, 3, 3, 4, 4, 5, 6])

        self.assertEqual(ecdf.confidence_interval(0.1, 0.9), (2., 6.))
        self.assertEqual(ecdf.confidence_interval(0.05, 0.95), (1., 6.))

        np.testing.assert_array_almost_equal(ecdf.heat_bar(6)[0], np.array([1., 6.]))
        np.testing.assert_array_almost_equal(ecdf.heat_bar(6)[1], np.array([0.12, 0.24, 0.36, 0.24, 0.12, 0.12]))

        self.assertEqual(ecdf.confidence_band_width(0.1, 0.9), (6. - 2.) / 6.)
