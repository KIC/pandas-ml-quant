from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from copy import deepcopy
from typing import List, Callable, Type, Dict, Tuple

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from torch.autograd import Variable

from pandas_ml_common import Typing, Sampler
from pandas_ml_common.utils import call_callable_dynamic_args, to_pandas
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.model.base_model import Model, AutoEncoderModel
from pandas_ml_utils.ml.summary import Summary
from pandas_ml_utils_torch.utils import from_pandas, FittingContext

_log = logging.getLogger(__name__)


class _PytorchModeBase(Model):
    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 module_provider: Type[PytorchNN],
                 criterion_provider: Type[nn.modules.loss._Loss],
                 optimizer_provider: Type[t.optim.Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 restore_best_weights: bool = False,
                 merge_cross_folds: Callable[[Dict[int, FittingContext.FoldContext]], Dict] = None,
                 callbacks: Dict[str, List[Callable]] = {},
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.module_provider = module_provider
        self.criterion_provider = criterion_provider
        self.optimizer_provider = optimizer_provider
        self.restore_best_weights = restore_best_weights
        self.merge_cross_folds = merge_cross_folds
        self.callbacks = callbacks
        self.module: PytorchNN = None
        self.log_once = LogOnce().log
        self._cuda = False
        self._fitting_context: FittingContext = None

    def cuda(self):
        self._cuda = True
        return self

    def cpu(self):
        self._cuda = False
        return self

    def init_fit(self, **kwargs):
        self._fitting_context = FittingContext(self.merge_cross_folds)

    def init_fold(self, epoch: int, fold: int):
        module = self.module_provider() if self.module is None else deepcopy(self.module)
        self._fitting_context.init_if_not_exists(fold, lambda: FittingContext.FoldContext(
            module,
            self.criterion_provider,
            self.optimizer_provider,
            self._cuda
        ))

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        cuda = self._cuda
        module, criterion, optimizer = self._fitting_context.get_module(fold)

        nnx = from_pandas(x, cuda)
        nny = from_pandas(y, cuda)
        weights = from_pandas(w, cuda, t.ones(nny.shape[0]))

        # ===================forward=====================
        output = module(nnx)
        loss = self._calc_weighted_loss(criterion, output, nny, weights)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def calculate_loss(self, fold, x, y_true, weight) -> float:
        cuda = self._cuda
        module = self.module.cuda() if cuda else self.module.cpu()
        ctx = FittingContext.FoldContext(module, self.criterion_provider, lambda p: None, self._cuda)

        with t.no_grad():
            y_pred = module(from_pandas(x, cuda))
            w = from_pandas(weight, cuda, t.ones(y_true.shape[0]))
            y_true = from_pandas(y_true, cuda)
            return self._calc_weighted_loss(ctx.criterion_l1_l2, y_pred, y_true, w).cpu().item()

    def _calc_weighted_loss(self, criterion, y_hat, y, weights):
        loss = criterion.loss_function(y_hat, y) + criterion.l1 + criterion.l2

        if loss.ndim > 0:
            if loss.ndim == weights.ndim:
                loss = (loss * weights).mean()
            else:
                self.log_once("loss.ndim!=weights.ndim", _log.warning,
                              f"sample weight has different dimensions {loss.shape}, {weights.shape}")

                if weights.ndim > loss.ndim and weights.shape[-1] == 1:
                    loss = self._calc_weighted_loss(criterion, y_hat, y, weights.squeeze())
                else:
                    loss = (loss * weights.repeat(1, *loss.shape[1:])).mean()

        return loss

    def after_fold_epoch(self, epoch, fold, fold_epoch, loss, val_loss):
        self._fitting_context.update_best_loss(fold, val_loss)

    def merge_folds(self, epoch: int):
        if self.restore_best_weights:
            self._fitting_context.restore_best_weights()

        self.module = self._fitting_context.merge_folds(self.module_provider, self._cuda)

    def _predict(self, features: pd.DataFrame, col_names, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        cuda = kwargs.get("cuda", False)
        module = self.module.cuda() if cuda else self.module.cpu()

        with t.no_grad():
            def predictor():
                return module(from_pandas(features, cuda)).cpu().numpy()

            y_hat = np.array([predictor() for _ in range(samples)]).swapaxes(0, 1) if samples > 1 else predictor()

        return to_pandas(y_hat, features.index, col_names)

    def finish_learning(self):
        self._fitting_context = None

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes.
        # Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()

        # remove un-pickleable fields
        del state['module']

        # add torch serialisation
        state['module_state_dict'] = self.module.state_dict() if self.module is not None else None

        # return altered state
        return state

    def __setstate__(self, state):
        # use torch.save(model.state_dict(), './sim_autoencoder.pth')
        # first remove the special state
        module_state_dict = state['module_state_dict']
        del state['module_state_dict']

        # Restore instance attributes
        self.__dict__.update(state)
        self.module = self.module_provider()

        # restore special state dict
        if module_state_dict is not None:
            self.module.load_state_dict(module_state_dict)


class PytorchModel(_PytorchModeBase):

    def __init__(self, features_and_labels: FeaturesAndLabels, module_provider: Type[PytorchNN],
                 criterion_provider: Type[nn.modules.loss._Loss], optimizer_provider: Type[t.optim.Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 restore_best_weights: bool = False,
                 merge_cross_folds: Callable[[Dict[int, FittingContext.FoldContext]], Dict] = None,
                 callbacks: Dict[str, List[Callable]] = {}, **kwargs):
        super().__init__(features_and_labels, module_provider, criterion_provider, optimizer_provider, summary_provider,
                         restore_best_weights, merge_cross_folds, callbacks, **kwargs)

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        self.module = self.module.eval()
        return self._predict(features, self._labels_columns, samples, **kwargs)


class PytorchAutoEncoderModel(_PytorchModeBase, AutoEncoderModel):

    def __init__(self, features_and_labels: FeaturesAndLabels, module_provider: Type[PytorchNN],
                 criterion_provider: Type[nn.modules.loss._Loss], optimizer_provider: Type[t.optim.Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 restore_best_weights: bool = False,
                 merge_cross_folds: Callable[[Dict[int, FittingContext.FoldContext]], Dict] = None,
                 callbacks: Dict[str, List[Callable]] = {}, **kwargs):
        super().__init__(features_and_labels, module_provider, criterion_provider, optimizer_provider, summary_provider,
                         restore_best_weights, merge_cross_folds, callbacks, **kwargs)

    def _auto_encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        self.module = self.module.eval()
        return self._predict(features, self._labels_columns, samples, **kwargs)

    def _encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        self.module = self.module.eval()
        return self._predict(features, self._features_and_labels.latent_names, samples, **kwargs)

    def _decode(self, latent_features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        self.module = self.module.eval()
        return self._predict(latent_features,  self._labels_columns, samples, **kwargs)


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
