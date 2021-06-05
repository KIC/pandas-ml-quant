import logging
from collections import defaultdict
from typing import Callable, Type

import pandas as pd
import tensorflow as tf

from pandas_ml_common import Typing
from pandas_ml_utils import FeaturesAndLabels, Summary

from pandas_ml_utils.ml.model.base_model import Model
from pandas_ml_utils_tf.tf_nn import TensorflowNN
from pandas_ml_utils_tf.utils import from_pandas

_log = logging.getLogger(__name__)


class _AbstractTFModel(Model):

    @staticmethod
    def trainable(provider, **kwargs):
        def wrapped():
            model = provider(**kwargs)
            return tf.function(model.forward_training), model.trainable_variables()
        return wrapped

    def __init__(self,
                 net_provider: Type[TensorflowNN],
                 features_and_labels: FeaturesAndLabels,
                 loss_function,
                 optimizer,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.net_provider = net_provider
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.folds = defaultdict(_AbstractTFModel.trainable(self.net_provider, **kwargs))
        self.net: TensorflowNN = None

    def init_fit(self, **kwargs):
        pass

    def init_fold(self, epoch: int, fold: int):
        self.net = self.folds[fold]

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, weight: pd.DataFrame, fold: int, **kwargs):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)  # quite high lr for normalizing flows.
        # gradients = _tf_fit(self.net, self.loss_function, from_pandas(x), from_pandas(y) )
        trainable, variables = self.folds[fold]
        with tf.GradientTape() as tape:
            loss = self.loss_function(from_pandas(y), trainable(from_pandas(x)))
            gradients = tape.gradient(loss, variables)

        #self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables()))

    def calculate_loss(self, fold, x, y_true, weight) -> float:
        pass

    def merge_folds(self, epoch: int):
        pass

    def finish_learning(self):
        pass

@tf.function()
def _tf_fit(model, loss_function, x, y):
    with tf.GradientTape() as tape:
        loss = loss_function(from_pandas(y), model.forward_training(from_pandas(x)))
        gradients = tape.gradient(loss, model.trainable_variables())

    return gradients

class TFModel(_AbstractTFModel):

    def __init__(self, net_provider: Type[TensorflowNN], features_and_labels: FeaturesAndLabels, loss_function,
                 optimizer, summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary, **kwargs):
        super().__init__(net_provider, features_and_labels, loss_function, optimizer, summary_provider, **kwargs)

