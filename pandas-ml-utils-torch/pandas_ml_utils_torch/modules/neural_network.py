import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Callable

import torch as t
from torch import nn

from pandas_ml_utils import AutoEncoderModel

_log = logging.getLogger(__name__)


class PytorchNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input, state=None, **kwargs):
        if self.training:
            return self.forward_training(*input, **kwargs)
        else:
            if state == AutoEncoderModel.ENCODE:
                return self.encode(*input, **kwargs)
            elif state == AutoEncoderModel.DECODE:
                return self.decode(*input, **kwargs)
            else:
                return self.forward_predict(*input, **kwargs)

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
            **kwargs) -> Callable[[], PytorchNN]:

        if predictor is None:
            predictor = lambda net, i: net(i)

        def init():
            return PytorchNNFactory(deepcopy(net), predictor, predictor if trainer is None else trainer, **kwargs)

        return init

    def __init__(self, net, predictor, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.predictor = predictor
        self.trainer = trainer

    def forward_training(self, *input) -> t.Tensor:
        return self.trainer(self.net, *input)

    def forward_predict(self, *input) -> t.Tensor:
        return self.predictor(self.net, *input)


class PytorchAutoEncoderFactory(PytorchNN):

    @staticmethod
    def create(
            encoder: nn.Module,
            decoder: nn.Module,
            **kwargs) -> Callable[[], PytorchNN]:

        def init():
            return PytorchNNFactory(deepcopy(encoder), deepcopy(decoder), **kwargs)

        return init

    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward_training(self, *input) -> t.Tensor:
        x = self.encoder(*input)
        x = self.decoder(x)
        return x

    def encode(self, *input) -> t.Tensor:
        return self.encoder(*input)

    def decode(self, *input) -> t.Tensor:
        return self.decoder(*input)
