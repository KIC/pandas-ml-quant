import logging
from copy import deepcopy

from wcmatch import glob
from typing import List, Union, Type, Callable, Iterable, Optional, Tuple
import torch as t
from torch import nn

from pandas_ml_common.utils import as_empty_tuple
from pandas_ml_common.utils.logging_utils import LogOnce
from ..modules import PytorchNN
from ..utils import to_device, from_pandas
from pandas_ml_common import np, MlTypes, XYWeight, call_callable_dynamic_args
from pandas_ml_utils import ModelProvider, FittingParameter, AutoEncoderModel

_log = logging.getLogger(__name__)
VarLenTensor = Union[t.Tensor, List[t.Tensor], Tuple[t.Tensor, ...]]


class PytorchModelProvider(ModelProvider):

    _MODEL_FIELD = "net"

    def __init__(self,
                 net: Union[Callable[..., PytorchNN], PytorchNN],
                 criterion_provider: Union[Type[nn.modules.loss._Loss], Callable[[], nn.modules.loss._Loss]],
                 optimizer_provider: Union[Type[t.optim.Optimizer], Callable[[Iterable], t.optim.Optimizer]],
                 record_best_weights: bool = False,
                 **kwargs):
        super().__init__()
        self.net_provider = _as_callable_wrapper(net)
        self.criterion_provider = criterion_provider
        self.optimizer_provider = optimizer_provider
        self.record_best_weights = record_best_weights
        self.kwargs = kwargs

        self.net = None
        self.criterion = None
        self.optimizer = None

        self.log_once = LogOnce().log
        self.best_weights = None
        self.best_loss = float('inf')
        self.last_loss = float('inf')
        self.l1_penalty_tensors = {t.zeros(1): 1.}
        self.l2_penalty_tensors = {t.zeros(1): 1.}

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        self.net = to_device(call_callable_dynamic_args(self.net_provider, **self.kwargs) if self.net is None else self.net, kwargs.get("cuda", False))
        self.criterion = to_device(call_callable_dynamic_args(self.criterion_provider, module=self.net, params=self.net.named_parameters()), kwargs.get("cuda", False))
        self.optimizer = self.optimizer_provider(self.net.parameters())

        # initialization
        self.optimizer.zero_grad()
        if hasattr(self.net, "init_weights"):
            self.net.apply(self.net.init_weights)

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

    def fit_batch(self, xyw: XYWeight, **kwargs):
        cuda = kwargs.get("cuda", False)
        x, y, sample_weight = from_pandas(xyw.x, cuda), from_pandas(xyw.y, cuda), from_pandas(xyw.weight, cuda)
        if not self.net.training:
            self.net = self.net.train()

        # forward
        output = self.net(*x)
        loss = self._calc_weighted_loss(self.criterion, output, y, sample_weight)

        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # loss and statistics
        loss_value = loss.cpu().item()
        self.last_loss = loss_value
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            if self.record_best_weights:
                self.best_weights = deepcopy(self.net.state_dict())

    def after_epoch(self, epoch, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        # if verbose we could print something
        pass

    def calculate_loss(self, xyw: XYWeight, **kwargs) -> float:
        cuda = kwargs.get("cuda", False)
        x, y, sample_weight = from_pandas(xyw.x, cuda), from_pandas(xyw.y, cuda), from_pandas(xyw.weight, cuda)
        return self._calculate_loss(x, y, sample_weight).cpu().item()

    def _calculate_loss(self, x: VarLenTensor, y_true: VarLenTensor, sample_weight: Optional[t.Tensor]) -> t.Tensor:
        with t.no_grad():
            y_pred = self.net(*x)
            return self._calc_weighted_loss(self.criterion, y_pred, y_true, sample_weight)

    def _calc_weighted_loss(self, criterion: nn.modules.loss._Loss, y_hat: VarLenTensor, y: VarLenTensor, weights: Optional[t.Tensor]) -> t.Tensor:
        l1 = t.stack([penalty * tensor.norm(p=1) for tensor, penalty in self.l1_penalty_tensors.items()]).sum()
        l2 = t.stack([penalty * tensor.norm(p=2) ** 2 for tensor, penalty in self.l2_penalty_tensors.items()]).sum()
        loss = criterion(*as_empty_tuple(y_hat), *as_empty_tuple(y)) + l1 + l2

        if loss.ndim > 0:
            if weights is None:
                weights = t.ones(y_hat.shape[0])

            if loss.ndim == weights.ndim:
                loss = (loss * weights).mean()
            elif weights.ndim > loss.ndim and weights.shape[-1] == 1:
                loss = (loss * weights.squeeze()).mean()
            else:
                self.log_once("loss.ndim!=weights.ndim", _log.warning, f"sample weight has different dimensions {loss.shape}, {weights.shape}")
                loss = (loss * weights.repeat(1, *loss.shape[1:])).mean()

        return loss

    def finish_learning(self, **kwargs):
        self.net = self.net.eval()
        self.criterion = None
        self.optimizer = None

    def train_predict(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        x = from_pandas(features, kwargs.get("cuda", False))
        with t.no_grad():
            return t.stack([self.net.forward_training(*x) for _ in range(samples)], dim=1) if samples > 1 else \
                self.net.forward_training(*x).cpu().numpy()

    def predict(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        x = from_pandas(features, kwargs.get("cuda", False))
        with t.no_grad():
            return t.stack([self.net(*x) for _ in range(samples)], dim=1) if samples > 1 else self.net(*x).cpu().numpy()

    def encode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        x = from_pandas(features, kwargs.get("cuda", False))
        with t.no_grad():
            return t.stack([self.net.forward(*x, state=AutoEncoderModel.ENCODE) for _ in range(samples)], dim=1) if samples > 1 else \
                self.net.forward(*x, state=AutoEncoderModel.ENCODE).cpu().numpy()

    def decode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        x = from_pandas(features, kwargs.get("cuda", False))
        with t.no_grad():
            return t.stack([self.net.forward(*x, state=AutoEncoderModel.DECODE) for _ in range(samples)], dim=1) if samples > 1 else \
                self.net.forward(*x, state=AutoEncoderModel.DECODE).cpu().numpy()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes.
        # Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()

        # remove un-pickleable fields
        del state[PytorchModelProvider._MODEL_FIELD]

        # add torch serialisation
        state[f'{PytorchModelProvider._MODEL_FIELD}_state_dict'] = self.net.state_dict() if self.net is not None else None

        # return altered state
        return state

    def __setstate__(self, state):
        # use torch.save(model.state_dict(), './sim_autoencoder.pth')
        # first remove the special state
        module_state_dict = state[f'{PytorchModelProvider._MODEL_FIELD}_state_dict']
        del state[f'{PytorchModelProvider._MODEL_FIELD}_state_dict']

        # Restore instance attributes
        self.__dict__.update(state)
        self.net = call_callable_dynamic_args(self.net_provider, **self.kwargs)

        # restore special state dict
        if module_state_dict is not None:
            self.net.load_state_dict(module_state_dict)

    def __call__(self, *args, **kwargs):
        copy = PytorchModelProvider(
            self.net_provider,
            self.criterion_provider,
            self.optimizer_provider,
            self.record_best_weights,
            **self.kwargs
        )

        # continue training from where we left
        if self.net is not None:
            copy.net = call_callable_dynamic_args(self.net_provider, **self.kwargs)
            copy.net.load_state_dict(self.net.state_dict())

        return copy


def _as_callable_wrapper(obj):
    if callable(obj) and not isinstance(obj, nn.Module):
        return obj
    else:
        def wrapper(*args, **kwargs):
            return obj

        return wrapper


