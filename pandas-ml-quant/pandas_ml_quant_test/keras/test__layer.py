import os
from unittest import TestCase

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import SGD

import pandas_ml_utils as pmu
from pandas_ml_quant import pd, np
from pandas_ml_quant.analysis.encoders import ReScaler
from pandas_ml_quant.keras.layers import LinearRegressionLayer, LPPLLayer, NormalDistributionLayer
from pandas_ml_quant.keras.loss import mdn_cost
from pandas_ml_quant_test.config import DF_TEST

os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(pmu.__version__)


class TestKerasLayer(TestCase):

    def test_normal_layer(self):
        """given"""
        df = DF_TEST.copy()
        model = Sequential()
        model.add(Dense(10))
        model.add(NormalDistributionLayer())
        model.compile(loss=mdn_cost, optimizer='adam')

        x = df["Close"].ta.rnn(range(5))._.values.squeeze()
        print(x.shape)

        # FIXME errors if bazch size is > 1
        model.fit(x, x, epochs=2, verbose=1, batch_size=1)

    def __test__LinearRegressionLayer(self):
        """given"""
        df = DF_TEST.copy()
        model = Sequential()
        model.add(LinearRegressionLayer())
        model.compile(loss='mse', optimizer='nadam')

        x = df["Close"].values.reshape(1, -1)

        """when"""
        model.fit(x, x, epochs=500, verbose=0)

        "then"
        res = pd.DataFrame({"close": df["Close"], "reg": model.predict_on_batch(x)}, index=df.index)
        print(res.head())

    def test__LPPLLayer(self):
        """given"""
        df = DF_TEST.copy()
        model = Sequential([LPPLLayer()])
        model.compile(loss='mse', optimizer=SGD(0.2, 0.01))
        #model.compile(loss='mse', optimizer='adam')

        x = np.log(df["Close"].values)
        x = ReScaler((x.min(), x.max()), (1, 2))(x)
        x = x.reshape(1, -1)

        x2 = np.vstack([x, x])

        """when"""
        model.fit(x2, x2, epochs=5000, verbose=0, callbacks=[EarlyStopping('loss')])

        """then"""
        print(model.predict_on_batch(x))
        res = pd.DataFrame({"close": x[0], "lppl": model.predict_on_batch(x)}, index=df.index)
        res.to_csv('/tmp/lppl.csv')
        print(res.head())
        print(model.layers[0].get_weights())