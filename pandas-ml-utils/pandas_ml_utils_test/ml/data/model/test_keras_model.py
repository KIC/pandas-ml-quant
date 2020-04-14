import os
from unittest import TestCase

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from pandas_ml_utils import KerasModel
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow import set_random_seed
set_random_seed(42)


class TestKerasModel(TestAbstractModel, TestCase):

    def provide_classification_model(self, features_and_labels):
        def model_provider():
            model = Sequential([
                Dense(3, input_shape=(2,), activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=SGD(lr=0.1), loss='mean_squared_error')
            return model

        model = KerasModel(
            model_provider,
            features_and_labels,
        )

        return model

    def provide_regression_model(self, features_and_labels):
        def model_provider():
            model = Sequential([
                Dense(units=1, input_shape=(1, ))
            ])

            model.compile(optimizer='sgd', loss='mean_squared_error')
            return model

        model = KerasModel(
            model_provider,
            features_and_labels,
        )

        return model

    def test_custom_object(self):
        # test safe and load
        pass


