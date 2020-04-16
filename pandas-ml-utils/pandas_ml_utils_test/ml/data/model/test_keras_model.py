import os
import tempfile
import uuid
from unittest import TestCase

import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from tensorflow import set_random_seed

from pandas_ml_utils import KerasModel, FeaturesAndLabels, Model
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel
from pandas_ml_common import pd

os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_random_seed(42)
np.random.seed(42)


def custom_loss_function(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred) * keras.backend.constant(2)


class TestKerasModel(TestAbstractModel, TestCase):

    def provide_classification_model(self, features_and_labels):
        def model_provider():
            model = Sequential([
                Dense(3, input_shape=(2,), activation='tanh'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=SGD(lr=0.1, momentum=0.9), loss='mean_squared_error')
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

    def test_custom_objects(self):
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        def model_provider():
            model = Sequential([
                Dense(units=1, input_shape=(1,))
            ])

            model.compile(optimizer='sgd', loss=custom_loss_function)
            return model, custom_loss_function

        model = KerasModel(
            model_provider,
            FeaturesAndLabels(["a"], ["b"]),
        )

        fit = df.model.fit(model)
        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        try:
            fit.model.save(temp)
            copy = Model.load(temp)
            pd.testing.assert_frame_equal(df.model.predict(fit.model), df.model.predict(copy), check_less_precise=True)
        finally:
            os.remove(temp)



