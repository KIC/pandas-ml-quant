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
            trainer: Callable[[nn.Module, t.Tensor], t.Tensor] = None) -> Callable[[], PytorchNN]:

        if predictor is None:
            predictor = lambda net, i: net(i)

        def factory(**kwargs):
            return PytorchNNFactory(net, predictor, predictor if trainer is None else trainer, **kwargs)

        return factory

    def __init__(self, net, predictor, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.predictor = predictor
        self.trainer = trainer

    def forward_training(self, *input) -> t.Tensor:
        return self.trainer(self.net, *input)

    def forward_predict(self, *input) -> t.Tensor:
        return self.predictor(self.net, *input)


class PytochBaseModel(object):

    _MODEL_FIELD = "net"

    def __init__(self,
                 net_provider: Union[Type[PytorchNN], Callable[[], PytorchNN]],
                 criterion_provider: Union[Type[nn.modules.loss._Loss], Callable[[], nn.modules.loss._Loss]],
                 optimizer_provider: Union[Type[t.optim.Optimizer], Callable[[Iterable], t.optim.Optimizer]],
                 record_best_weights: bool = False,
                 cuda: bool = False,
                 **kwargs
                 ):

        self.net_provider = net_provider
        self.criterion_provider = criterion_provider
        self.optimizer_provider = optimizer_provider
        self._cuda = cuda

        self.net = to_device(net_provider(**kwargs), cuda)
        self.criterion = to_device(call_callable_dynamic_args(criterion_provider, module=self.net, params=self.net.named_parameters()), cuda)
        self.optimizer = optimizer_provider(self.net.parameters())
        self.log_once = LogOnce().log
        self.best_weights = None
        self.best_loss = float('inf')
        self.last_loss = float('inf')
        self.record_best_weights = record_best_weights

        # initialization
        self.optimizer.zero_grad()
        if hasattr(self.net, "init_weights"):
            self.net.apply(self.net.init_weights)

        # penalty
        self.l1_penalty_tensors = {t.zeros(1): 1.}
        self.l2_penalty_tensors = {t.zeros(1): 1.}

        def param_dict(module):
            return {t[0].replace('.', '/'): t[1] for t in module.named_parameters()}

        if hasattr(self.net, "L1") and len(self.net.L1()) > 0:
            self.l1_penalty_tensors = \
                {tensor: penalty for param, tensor in param_dict(self.net).items()
                 for path, penalty in self.net.L1().items()
                 if glob.globmatch(param, path, flags=glob.GLOBSTAR)}

        if hasattr(self.net, "L2") and len(self.net.L2()) > 0:
            self.l2_penalty_tensors = \
                {tensor: penalty for param, tensor in param_dict(self.net).items()
                 for path, penalty in self.net.L2().items()
                 if glob.globmatch(param, path, flags=glob.GLOBSTAR)}

    def cpu(self):
        self.net = self.net.cpu()
        self.criterion = self.criterion.cpu()
        return self

    def cuda(self):
        self.net = self.net.cuda()
        self.criterion = self.criterion.cuda()
        return self

    def fit_epoch(self, x: t.Tensor, y: t.Tensor, sample_weight: t.Tensor = None):
        if not self.net.training:
            self.net.train()

        # ===================forward=====================
        output = self.net(x)
        loss = self._calc_weighted_loss(self.criterion, output, y, sample_weight)

        # ===================backward====================
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #
        loss_value = loss.cpu().item()
        self.last_loss = loss_value
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            if self.record_best_weights:
                self.best_weights = deepcopy(self.net.state_dict())

        # print('lala', loss_value)
        return loss_value

    def calculate_loss(self, x: t.Tensor, y_true: t.Tensor, sample_weight: t.Tensor):
        with t.no_grad():
            y_pred = self.net(x)
            return self._calc_weighted_loss(self.criterion, y_pred, y_true, sample_weight).cpu().item()

    def _calc_weighted_loss(self, criterion: nn.modules.loss._Loss, y_hat: t.Tensor, y: t.Tensor, weights: t.Tensor):
        l1 = t.stack([penalty * tensor.norm(p=1) for tensor, penalty in self.l1_penalty_tensors.items()]).sum()
        l2 = t.stack([penalty * tensor.norm(p=2) ** 2 for tensor, penalty in self.l2_penalty_tensors.items()]).sum()
        loss = criterion(y_hat, y) + l1 + l2

        if loss.ndim > 0:
            if weights is None:
                weights = t.ones(y_hat.shape[0])

            if loss.ndim == weights.ndim:
                loss = (loss * weights).mean()
            elif weights.ndim > loss.ndim and weights.shape[-1] == 1:
                loss = (loss * weights.squeeze()).mean()
            else:
                self.log_once("loss.ndim!=weights.ndim", _log.warning,
                              f"sample weight has different dimensions {loss.shape}, {weights.shape}")

                loss = (loss * weights.repeat(1, *loss.shape[1:])).mean()

        return loss

    def restore_best_weights(self):
        if self.best_weights is not None:
            self.net.load_state_dict(self.best_weights)
        else:
            _log.warning("No best weights found!! Keep existing weights.")

    def predict(self, x: t.Tensor, samples=1, numpy=True, force_mode: bool = False) -> Union[np.ndarray, t.Tensor]:
        if force_mode and self.net.training:
            self.net.eval()

        with t.no_grad():
                y_hat = t.stack([self.net(x) for _ in range(samples)], dim=1) if samples > 1 else self.net(x)

        return y_hat.cpu().numpy() if numpy else y_hat

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def shadow_copy(self):
        pbm = PytochBaseModel(
            lambda: deepcopy(self.net),
            self.criterion_provider,
            self.optimizer_provider,
            self.record_best_weights,
            self._cuda
        )

        pbm.log_once = self.log_once
        return pbm

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes.
        # Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()

        # remove un-pickleable fields
        del state[PytochBaseModel._MODEL_FIELD]

        # add torch serialisation
        state[f'{PytochBaseModel._MODEL_FIELD}_state_dict'] = self.net.state_dict() if self.net is not None else None

        # return altered state
        return state

    def __setstate__(self, state):
        # use torch.save(model.state_dict(), './sim_autoencoder.pth')
        # first remove the special state
        module_state_dict = state[f'{PytochBaseModel._MODEL_FIELD}_state_dict']
        del state[f'{PytochBaseModel._MODEL_FIELD}_state_dict']

        # Restore instance attributes
        self.__dict__.update(state)
        self.net = self.net_provider()

        # restore special state dict
        if module_state_dict is not None:
            self.net.load_state_dict(module_state_dict)

        # restore optimizer and loss, this allows us to continue training
        self.optimizer = self.optimizer_provider(self.net.parameters())
        self.criterion = to_device(
            call_callable_dynamic_args(self.criterion_provider, module=self.net, params=self.net.named_parameters()),
            False
        )
