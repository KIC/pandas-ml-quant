import os
from unittest import TestCase

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from pandas_ml_common import pd
from pandas_ml_utils import KerasModel, FeaturesAndLabels
from pandas_ml_utils.ml.data.splitting import RandomSplits
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestKerasModel(TestAbstractModel, TestCase):

    def provide_regression_model(self):
        def model_provider():
            model = Sequential([
                Dense(units=1, input_shape=(1, ))
            ])

            model.compile(optimizer='sgd', loss='mean_squared_error')
            return model

        model = KerasModel(
            model_provider,
            FeaturesAndLabels(features=["a"], labels=["b"]),
        )

        return model

    def test_custom_object(self):
        # test safe and load
        pass


