from unittest import TestCase
import numpy as np

from pandas_ml_common.utils import normalize_probabilities


class TestRandumUtils(TestCase):

    def test_probs_normalisation(self):
        p1 = [0.2, 0.8]
        p2 = [2, 8]

        np.testing.assert_equal(np.array(p1), normalize_probabilities(p1))
        np.testing.assert_equal(np.array(p1), normalize_probabilities(p2))
