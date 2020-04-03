import os
from unittest import TestCase

from keras import Sequential
from keras.layers import Dense, LSTM

import pandas_ml_utils as pmu
from pandas_ml_quant.keras.time_to_vec import Time2Vec
from pandas_ml_quant_test.config import DF_TEST

os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(pmu.__version__)


class TestTime2Vec(TestCase):

    def test_time_to_vec(self):
        df = DF_TEST.copy()
        model = Sequential()
        model.add(Time2Vec(32, input_shape=(5, 1)))  # shape (?, 5, 32)
        model.add(LSTM(32))  # shape (?, 160)
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        x = df["Volume"].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        model.fit(x, y, epochs=2, verbose=1)