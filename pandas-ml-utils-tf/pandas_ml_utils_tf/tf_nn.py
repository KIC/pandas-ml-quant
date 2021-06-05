from abc import abstractmethod
from typing import List, Union, Callable

import tensorflow as tf
from keras import Model
from tensorflow.python.ops.variables import VariableMetaclass
from tensorflow.python.training.tracking import base as trackable


class TensorflowNN(object):

    # we will later explicitly wrap this function into `@tf.function`
    # The Function stores the tf.Graph corresponding to that signature in a ConcreteFunction.
    # A ConcreteFunction is a wrapper around a tf.Graph.
    def forward_training(self, *input) -> tf.Tensor:
        return self.forward_predict(*input)

    # we will later explicitly wrap this function into `@tf.function`
    # The Function stores the tf.Graph corresponding to that signature in a ConcreteFunction.
    # A ConcreteFunction is a wrapper around a tf.Graph.
    @abstractmethod
    def forward_predict(self, *input) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def trainable_variables(self) -> List[Union[VariableMetaclass, trackable.Trackable]]:
        raise NotImplementedError


class TensorflowNNFactory(TensorflowNN):

    @staticmethod
    def create(
            net: Model,
            predictor: Callable[[Model, tf.Tensor], tf.Tensor],
            trainer: Callable[[Model, tf.Tensor], tf.Tensor] = None):
        def factory(**kwargs):
            return TensorflowNNFactory(net, predictor, predictor if trainer is None else trainer, **kwargs)

        return factory

    def __init__(self, net, predictor, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.predictor = predictor
        self.trainer = trainer

    def forward_training(self, *input) -> tf.Tensor:
        return self.trainer(self.net, *input)

    def forward_predict(self, *input) -> tf.Tensor:
        return self.predictor(self.net, *input)

    def trainable_variables(self) -> List[Union[VariableMetaclass, trackable.Trackable]]:
        return self.net.trainable_variables

