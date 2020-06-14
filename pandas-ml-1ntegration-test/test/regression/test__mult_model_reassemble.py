import os
from unittest import TestCase

from pandas_ml_quant import pd
from pandas_ml_utils import Model
from test.config import TEST_DATA_PATH


class TestMultiModelReAssembling(TestCase):

    def test_reassembling(self):
        df = pd.read_pickle(os.path.join(TEST_DATA_PATH, "btc_hourly.df"))
        model = Model.load(os.path.join(TEST_DATA_PATH, "MultiModel-Predictive-VAE--BTC.model"))

        prediction = df.model.predict(model, tail=2)
        print(prediction)

