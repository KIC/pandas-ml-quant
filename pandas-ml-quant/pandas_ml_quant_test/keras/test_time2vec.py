import os
from unittest import TestCase

from keras import Sequential
from keras.layers import Dense, LSTM

import pandas_ml_utils as pmu
from pandas_ml_quant.keras.time_2_vec import Time2Vec, tf_Time2Vec
from pandas_ml_quant_test.config import DF_TEST
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(pmu.__version__)


class TestTime2Vec(TestCase):

    def test_time_to_vec(self):
        df = DF_TEST.copy()
        model = Sequential()
        model.add(Time2Vec(32, input_shape=(5, 1)))  # -> shape (?, 5, 33)
        model.add(LSTM(32))  # shape (?, 160)
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        x = df["Volume"].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        model.fit(x, y, epochs=2, verbose=1)

    def test_time_to_vec_3d(self):
        # FIXME
        df = DF_TEST.copy()
        model = Sequential()
        model.add(Time2Vec(32, input_shape=(5, 1)))   # -> shape (?, 5, 33) but should be (?, 5, 33 * 2)
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        x = df[["Volume", "Volume"]].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        model.fit(x, y, epochs=2, verbose=1)

    def test_tf_time_to_vec_1d(self):
        df = DF_TEST.copy()
        model = tf.keras.models.Sequential()
        model.add(tf_Time2Vec(32, input_shape=(5, 1)))  # shape (?, 5, 32)
        model.add(tf.keras.layers.LSTM(32))  # shape (?, 160)
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')

        x = df["Volume"].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        model.fit(x, y, epochs=2, verbose=1)

    def test_tf_time_to_vec(self):
        df = DF_TEST.copy()
        model = tf.keras.models.Sequential()
        model.add(tf_Time2Vec(32, features=2, input_shape=(5, 2)))  # shape (?, 5, 32)
        model.add(tf.keras.layers.LSTM(32))  # shape (?, 160)
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')

        x = df[["Volume", "Volume"]].pct_change().ta.rnn(range(5))._.values[:-2]
        y = df["Volume"].pct_change().shift(-1).dropna()[6:]

        model.fit(x, y, epochs=2, verbose=1)