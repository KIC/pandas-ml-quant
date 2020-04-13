from unittest import TestCase

import os
from keras.layers import Dense, Input, LSTM, Lambda, Concatenate, Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import backend as K
from pandas_ml_common import pd
from pandas_ml_utils import KerasModel, FeaturesAndLabels

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestKerasModel(TestCase):

    def test_classifier(self):
        pass

    def test_regressor(self):
        pass

    def test_2D_auto_encoder(self):
        df = pd.DataFrame({"a": [
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        ]})

        def model_provider():
            m = Sequential()

            m.add(Conv2D(1, 1, input_shape=(1, 3, 3), data_format='channels_first', padding='same'))
            m.add(Activation('relu'))

            m.add(Conv2D(1, 1, data_format='channels_first', padding='same'))
            m.add(Activation('relu'))

            m.compile(loss='binary_crossentropy', optimizer='adadelta')
            return m

        model_provider().summary()

        fit = df.model.fit(
            KerasModel(
                model_provider,
                FeaturesAndLabels(
                    features=["a"],
                    labels=["a"]
                ),
                output_shape=(1, 3, 3)
            ),
            batch_size=2,
            epochs=20,
            verbose=1
        )

        fit
        print(fit)

    def test_custom_object(self):
        pass

