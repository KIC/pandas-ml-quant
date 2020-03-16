from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import uuid
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

from pandas_ml_common.utils import merge_kwargs, suitable_kwargs
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class KerasModel(Model):
    # eventually we need to save and load the weights of the keras model individually by using `__getstate__`
    #  `__setstate__` like described here: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    if TYPE_CHECKING:
        from keras.models import Model as KModel

    def __init__(self,
                 keras_compiled_model_provider: Callable[[], KModel],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 epochs: int = 100,
                 callbacks: List[Callable] = [],
                 **kwargs):
        """
        Keras compatible implementation of :class:`.Model`.
        :param keras_compiled_model_provider: a callable which provides an eventually compiled ready to fit keras model.
               if the model is not compiled you ned to pass "optimizer" argument as kwargs.
               NOTE: the tensorflow backend is currently limited to 1.*
        :param features_and_labels: see :class:`.Model`
        :param summary_provider: :class:`.Model`.
        :param epochs: number of epochs passed to the keras fit function
        :param callbacks: a list of callable's providing a keras compatible callback. It is neccessary to pass the
                          callable i.e. BaseLogger or lambda: BaseLogger(...) vs. BaseLogger(...)
        :param kwargs: a list of arguments passed to the keras_compiled_model_provider and keras fit function as they
                       match the signature
        """
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.keras_model_provider = keras_compiled_model_provider
        self.custom_objects = {}

        import keras
        if keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            self.is_tensorflow = True
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.session = tf.Session(graph=self.graph)
        else:
            self.is_tensorflow = False

        # create keras model
        provider_args = suitable_kwargs(keras_compiled_model_provider, **kwargs)
        keras_model = self._exec_within_session(keras_compiled_model_provider, **provider_args)

        if isinstance(keras_model, Tuple):
            # store custom objects
            for i in range(1, len(keras_model)):
                if hasattr(keras_model[i], "__name__"):
                    self.custom_objects[keras_model[i].__name__] = keras_model[i]
                else:
                    raise ValueError("keras custom object must have a __name__")

            keras_model = keras_model[0]
            _log.warning(f"keras model using custom objects {self.custom_objects}")

        # eventually compile keras model
        if not keras_model.optimizer:
            compile_args = suitable_kwargs(keras_model.compile, **kwargs)
            self._exec_within_session(keras_model.compile, **compile_args)

        # set all members
        self.keras_model = keras_model
        self.epochs = epochs
        self.callbacks = callbacks
        self.history = None

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray,
            sample_weight_train: np.ndarray, sample_weight_test: np.ndarray) -> float:
        fitter_args = suitable_kwargs(self.keras_model.fit, **self.kwargs)

        if sample_weight_train is not None:
            print(f"using sample weights {sample_weight_train.shape}")

        if len(fitter_args) > 0:
            print(f'pass args to fit: {fitter_args}')

        fit_history = self._exec_within_session(self.keras_model.fit,
                                                x, y,
                                                sample_weight=sample_weight_train,
                                                epochs=self.epochs,
                                                validation_data=(x_val, y_val),
                                                callbacks=[cb() for cb in self.callbacks],
                                                **fitter_args)
        if self.history is None:
            self.history = fit_history.history
        else:
            for metric, _ in self.history.items():
                self.history[metric] = self.history[metric] + fit_history.history[metric]

        return min(fit_history.history['loss'])

    def predict(self, x):
        return self._exec_within_session(self.keras_model.predict, x)

    def get_weights(self):
        return self._exec_within_session(self.keras_model.get_weights)

    def set_weights(self, weights):
        self._exec_within_session(self.keras_model.set_weights, weights)

    def plot_loss(self):
        import matplotlib.pyplot as plt

        plt.plot(self.history['val_loss'], label='test')
        plt.plot(self.history['loss'], label='train')
        plt.legend(loc='best')

    def _exec_within_session(self, func, *args, **kwargs):
        if self.is_tensorflow:
            with self.graph.as_default():
                with self.session.as_default():
                    return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes.
        # Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()

        # remove un-pickleable fields
        if self.is_tensorflow:
            del state['graph']
            del state['session']

        # special treatment for the keras model
        tmp_keras_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        self._exec_within_session(lambda: self.keras_model.save(tmp_keras_file, True, True))
        with open(tmp_keras_file, mode='rb') as file:
            state['keras_model'] = file.read()

        # clean up temp file
        with contextlib.suppress(OSError):
            os.remove(tmp_keras_file)

        # return state
        return state

    def __setstate__(self, state):
        from keras.models import load_model

        # restore keras file
        tmp_keras_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        with open(tmp_keras_file, mode='wb') as file:
            file.write(state['keras_model'])
            del state['keras_model']

        # Restore instance attributes
        self.__dict__.update(state)

        # restore keras model and tensorflow session if needed
        if self.is_tensorflow:
            from keras import backend as K
            import tensorflow as tf
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.session = tf.Session(graph=self.graph)
                K.set_session(self.session)
                self.keras_model = load_model(tmp_keras_file, custom_objects=self.custom_objects)
        else:
            self.keras_model = load_model(tmp_keras_file, custom_objects=self.custom_objects)

        # clean up temp file
        with contextlib.suppress(OSError):
            os.remove(tmp_keras_file)

    def __del__(self):
        if self.is_tensorflow:
            try:
                self.session.close()
            except AttributeError:
                pass

    def __call__(self, *args, **kwargs):
        new_model = KerasModel(self.keras_model_provider,
                               self.features_and_labels,
                               self.summary_provider,
                               self.epochs,
                               deepcopy(self.callbacks),
                               **merge_kwargs(deepcopy(self.kwargs), kwargs))

        # copy weights before return
        new_model.set_weights(self.get_weights())
        return new_model

