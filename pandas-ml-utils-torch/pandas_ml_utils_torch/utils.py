import sys
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Dict, Tuple, Union

import pandas as pd
import torch as t
from wcmatch import glob

from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.utils import unpack_nested_arrays, call_callable_dynamic_args


def from_pandas(df: pd.DataFrame, cuda: bool = False, default: t.Tensor = None) -> Union[t.Tensor, Tuple[t.Tensor, ...]]:
    if df is None and default is None:
        return None

    if isinstance(df, MultiFrameDecorator):
        return tuple([from_pandas(f, cuda, default) for f in df.frames()])

    val = (t.from_numpy(unpack_nested_arrays(df, split_multi_index_rows=False)) if df is not None else default).float()
    return val.cuda() if cuda else val.cpu()


def param_dict(module):
    return {t[0].replace('.', '/'): t[1] for t in module.named_parameters()}


class FittingContext(object):

    class FoldContext(object):

        Criterion = namedtuple("Criterion", ["loss_function", "l1", "l2"])

        def __init__(
                self,
                module: t.nn.Module,
                criterion: Callable[[], t.nn.modules.loss._Loss],
                optimizer: Callable[[], t.optim.Optimizer],
                cuda: bool
        ):
            self.module = module.cuda() if cuda else module.cpu()
            criterion = call_callable_dynamic_args(criterion, module=self.module, params=self.module.named_parameters())
            self.criterion = criterion.cuda() if cuda else criterion.cpu()
            self.optimizer = optimizer(self.module.parameters())
            self.best_weights = None
            self.best_loss = sys.float_info.max
            self.last_loss = sys.float_info.max
            self.l1_penalty_tensors = None
            self.l2_penalty_tensors = None
            self.cuda = cuda

            if hasattr(module, "L1") and len(module.L1()) > 0:
                self.l1_penalty_tensors = \
                    {tensor: penalty for param, tensor in param_dict(module).items()
                     for path, penalty in self.module.L1().items()
                     if glob.globmatch(param, path, flags=glob.GLOBSTAR)}

            if hasattr(module, "L2") and len(module.L2()) > 0:
                self.l2_penalty_tensors = \
                    {tensor: penalty for param, tensor in param_dict(module).items()
                     for path, penalty in self.module.L2().items()
                     if glob.globmatch(param, path, flags=glob.GLOBSTAR)}

        def update_best_loss(self, loss):
            self.last_loss = loss
            if loss < self.best_loss:
                self.best_loss = loss
                self.update_best_weights()

        def update_best_weights(self):
            self.best_weights = deepcopy(self.module.state_dict())

        def restore_best_weights(self):
            if self.best_weights is not None:
                self.module.load_state_dict(self.best_weights)
                self.last_loss = self.best_loss

        def get_l1_term(self):
            if self.l1_penalty_tensors is None:
                return t.zeros(1)

            return t.stack([penalty * tensor.norm(p=1) for tensor, penalty in self.l1_penalty_tensors.items()]).sum()

        def get_l2_term(self):
            if self.l2_penalty_tensors is None:
                return t.zeros(1)

            return t.stack([penalty * tensor.norm(p=2) ** 2 for tensor, penalty in self.l2_penalty_tensors.items()]).sum()

        @property
        def criterion_l1_l2(self):
            return FittingContext.FoldContext.Criterion(self.criterion, self.get_l1_term(), self.get_l2_term())

    def __init__(self, merge_cross_folds: Callable[[Dict[int, FoldContext]], Dict] = None,):
        self.merge_cross_folds = merge_cross_folds
        self.fold_modules: Dict[int, FittingContext.FoldContext] = {}

    def init_if_not_exists(self, fold: int, module):
        fold = self._translate_fold(fold)
        if fold not in self.fold_modules:
            self.fold_modules[fold] = module

    def update_best_loss(self, fold, loss):
        fold = self._translate_fold(fold)
        self.fold_modules[fold].update_best_loss(loss)

    def restore_best_weights(self):
        for m in self.fold_modules.values():
            m.restore_best_weights()

    def get_module(self, fold):
        fc = self.fold_modules[fold]
        return fc.module, fc.criterion_l1_l2, fc.optimizer

    def merge_folds(self, module_provider, cuda):
        merged_model = None
        if self.merge_cross_folds is None:
            for ctx in self.fold_modules.values():
                merged_model = ctx.module
                break
        else:
            module = module_provider()
            module = module.cuda() if cuda else module.cpu()
            module.load_state_dict(self.merge_cross_folds(self.fold_modules))
            merged_model = module

        self.fold_modules.clear()
        return merged_model

    def _translate_fold(self, fold):
        return 0 if self.merge_cross_folds is None else fold