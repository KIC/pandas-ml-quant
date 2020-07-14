from __future__ import annotations

import logging
import math
import sys
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Type, Dict, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class PytorchModel(Model):

    if TYPE_CHECKING:
        from torch.nn import Module, _Loss
        from torch.optim import Optimizer

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 module_provider: Type[Module],
                 criterion_provider: Type[_Loss],
                 optimizer_provider: Type[Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 callbacks: Dict[str, List[Callable]] = {},
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.module_provider = module_provider
        self.criterion_provider = criterion_provider
        self.optimizer_provider = optimizer_provider
        self.callbacks = callbacks
        self.module = None
        self.log_once = LogOnce().log

    def fit_fold(self,
                 fold_nr: int,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight: np.ndarray, sample_weight_val: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        # import specifics
        from torch.autograd import Variable
        import torch as t

        # TODO we should not re-initialize model, criterion and optimizer once we have it already
        #  TODO we might re-initialize the optimizer with a new fold with a changes learning rate?

        is_verbose = kwargs["verbose"] if "verbose" in kwargs else False
        on_epoch_callbacks = kwargs["on_epoch"] if "on_epoch" in kwargs else []
        restore_best_weights = kwargs["restore_best_weights"] if "restore_best_weights" in kwargs else False
        num_epochs = kwargs["epochs"] if "epochs" in kwargs else 100
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 128
        use_cuda = kwargs["cuda"] if "cuda" in kwargs else False

        module = self.module
        criterion_provider = self.criterion_provider

        if use_cuda:
            criterion_provider = lambda: self.criterion_provider().cuda()
            module = module.cuda()

        module = module.train()
        criterion = criterion_provider()
        optimizer = self.optimizer_provider(module.parameters())
        best_model_wts = deepcopy(module.state_dict())
        best_loss = sys.float_info.max
        epoch_losses = []
        epoch_val_losses = []

        if hasattr(module, 'callback'):
            on_epoch_callbacks += [module.callback]

        if hasattr(criterion, 'callback'):
            on_epoch_callbacks += [criterion.callback]

        if is_verbose:
            print(f"fit fold {fold_nr} with {len(x)} samples in {math.ceil(len(x) / batch_size)} batches ... ")

        for epoch in range(num_epochs):
            for i in range(0, len(x), batch_size):
                nnx = Variable(t.from_numpy(x[i:i+batch_size])).float()
                nny = Variable(t.from_numpy(y[i:i+batch_size])).float()
                weights = Variable(t.from_numpy(sample_weight[i:i + batch_size])).float() \
                    if sample_weight is not None else t.ones(nny.shape[0])

                if use_cuda:
                    nnx, nny, weights = nnx.cuda(), nny.cuda(), weights.cuda()

                if nnx.shape[0] <= 1:
                    self.log_once("invalid_batch", _log.warning, "skip single element batch!")
                    continue

                # ===================forward=====================
                output = module(nnx)
                loss = self._calc_weighted_loss(criterion, output, nny, weights)

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if is_verbose > 1:
                    print(f"{epoch}:{i}\t{loss}\t")

            # ===================log========================
            # add validation loss history
            if y_val is not None and len(y_val) > 0:
                with t.no_grad():
                    nnx = t.from_numpy(x).float()
                    nny = t.from_numpy(y).float()
                    weights = t.from_numpy(sample_weight).float() \
                        if sample_weight is not None else t.ones(nny.shape[0])
                    nnx_val = t.from_numpy(x_val).float()
                    nny_val = t.from_numpy(y_val).float()
                    weights_val = t.from_numpy(sample_weight_val).float() \
                        if sample_weight_val is not None else t.ones(nny_val.shape[0])

                    if use_cuda:
                        nnx, nny = nnx.cuda(), nny.cuda()
                        nnx_val, nny_val = nnx_val.cuda(), nny_val.cuda()
                        weights, weights_val = weights.cuda(), weights_val.cuda()

                    y_hat = module(nnx)
                    loss = self._calc_weighted_loss(criterion_provider(), y_hat, nny, weights).item()
                    epoch_losses.append(loss)

                    y_hat_val = module(nnx_val)
                    val_loss = self._calc_weighted_loss(criterion_provider(), y_hat_val, nny_val, weights_val).item()
                    epoch_val_losses.append(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_wts = deepcopy(module.state_dict())

            if is_verbose:
                print(f"{epoch}\t{loss}\t{val_loss}")

            # invoke on epoch end callbacks
            try:
                for callback in on_epoch_callbacks:
                    call_callable_dynamic_args(callback,
                                               fold=fold_nr,
                                               epoch=epoch,
                                               x=x, y=y, x_val=x_val, y_val=y_val,
                                               y_hat=y_hat, y_hat_val=y_hat_val,
                                               loss=loss, val_loss=val_loss, best_loss=best_loss)
            except StopIteration:
                break

        if restore_best_weights:
            module.load_state_dict(best_model_wts)

        return np.array(epoch_losses), np.array(epoch_val_losses)

    def _calc_weighted_loss(self, criterion, y_hat, y, weights):
        loss = criterion(y_hat, y)

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

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # import specifics
        import torch as t

        with t.no_grad():
            module = self.module.cpu().eval()
            res = module(t.from_numpy(x).float())
            return res if isinstance(res, np.ndarray) else res.numpy()

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

    def __call__(self, *args, **kwargs):
        pytorch_model = PytorchModel(
            self.features_and_labels,
            self.module_provider,
            self.criterion_provider,
            self.optimizer_provider,
            self.summary_provider,
            deepcopy(self.callbacks)
        )

        pytorch_model.module = self.module_provider()

        # copy weights of existing models
        if self.module is not None:
            pytorch_model.module.load_state_dict(deepcopy(self.module.state_dict()))

        return pytorch_model

    # Add some useful callbacks directly to the pytorch model
    class Callbacks(object):

        @staticmethod
        def print_loss(mod=10):
            def printer(fold, epoch, loss, val_loss):
                if epoch % mod == 0:
                    print(f"{fold}: {epoch}: {loss}\t {val_loss}")

            return printer

        class early_stopping(object):

            def __init__(self, patience=1, tolerance=0.001):
                self.patience = patience
                self.tolerance = tolerance
                self.last_loss = sys.float_info.max
                self.counter = 0

            def __call__(self, val_loss, **kwargs):
                if (val_loss - self.tolerance) < self.last_loss:
                    self.last_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f"early stopping {self.counter}, {val_loss} > {self.last_loss}")
                        raise StopIteration("early stopping")

            def __copy__(self):
                return type(self)(self.patience, self.tolerance)
