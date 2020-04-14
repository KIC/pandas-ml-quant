from unittest import TestCase

import os
from keras.layers import Dense, Input, LSTM, Lambda, Concatenate, Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model as KModel
from keras import backend as K
from pandas_ml_common import pd
from pandas_ml_utils import KerasModel, AutoEncoderModel, FeaturesAndLabels

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestKerasModel(TestCase):

    def test_classifier(self):
        # test safe and load
        pass

    def test_regressor(self):
        # test safe and load
        pass

    def test_custom_object(self):
        # test safe and load
        pass


