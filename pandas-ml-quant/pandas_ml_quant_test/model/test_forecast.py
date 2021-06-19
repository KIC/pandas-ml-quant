from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant.model.forecast.distribution_forecast import DistributionForecast
from pandas_ml_utils.constants import *


class TestForecast(TestCase):

    def test_distribution_forecast(self):
        df = pd.DataFrame({
            (TARGET_COLUMN_NAME, 'Close'): np.linspace(0, 1, 10),
            (PREDICTION_COLUMN_NAME, 'Close'): [[i, np.random.random(1)] for i in np.linspace(0, 1, 10)],
            (FEATURE_COLUMN_NAME, 'Close'): np.linspace(0, 1, 10),
        })

        fc = DistributionForecast(df, lambda param, samples: np.random.normal(param[0], param[1], samples))
        print(fc.hist())
        print(fc.df)