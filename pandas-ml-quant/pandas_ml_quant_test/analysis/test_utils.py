from unittest import TestCase
import pandas as pd
import numpy as np

from pandas_ml_quant.analysis.utils import conditional_func


class TestAnalysisUtils(TestCase):

    def test_conditional_func(self):
        s = pd.Series(np.linspace(0, 1, 10))
        res = conditional_func(s >= 0.5, s * -10, s * 10)

        print(res)

