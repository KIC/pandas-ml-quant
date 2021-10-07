import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Type, Callable, Union, Iterable, Tuple
from wcmatch import glob

import numpy as np
import torch as t
from torch import nn

from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils import AutoEncoderModel, call_callable_dynamic_args
from pandas_ml_utils_torch.utils import to_device

_log = logging.getLogger(__name__)


class PytorchNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input, state=None, **kwargs):
        if self.training:
            return self.forward_training(*input)
        else:
            if state == AutoEncoderModel.ENCODE:
                return self.encode(*input)
            elif state == AutoEncoderModel.DECODE:
                return self.decode(*input)
            else:
                return self.forward_predict(*input)

    @abstractmethod
    def forward_training(self, *input) -> t.Tensor:
        pass

    def forward_predict(self, *input) -> t.Tensor:
        return self.forward_training(*input)

    def encode(self, *input) -> t.Tensor:
        raise NotImplementedError("For autoencoders the methods `encode` and `decode` need to be implemented!")

    def decode(self, *input) -> t.Tensor:
        raise NotImplementedError("For autoencoders the methods `encode` and `decode` need to be implemented!")

    def L1(self) -> Dict[str, float]:
        return {}

    def L2(self) -> Dict[str, float]:
        return {}


class PytorchNNFactory(PytorchNN):

    @staticmethod
    def create(
            net: nn.Module,
            predictor: Callable[[nn.Module, t.Tensor], t.Tensor] = None,
            trainer: Callable[[nn.Module, t.Tensor], t.Tensor] = None,
            **kwargs) -> PytorchNN:

        if predictor is None:
            predictor = lambda net, i: net(i)

        return PytorchNNFactory(net, predictor, predictor if trainer is None else trainer, **kwargs)

    def __init__(self, net, predictor, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.predictor = predictor
        self.trainer = trainer

    def forward_training(self, *input) -> t.Tensor:
        return self.trainer(self.net, *input)

    def forward_predict(self, *input) -> t.Tensor:
        return self.predictor(self.net, *input)


