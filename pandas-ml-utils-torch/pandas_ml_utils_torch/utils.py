import sys
from copy import deepcopy
from typing import Callable, Tuple, Dict

import pandas as pd
import numpy as np
import torch as t

from pandas_ml_common.utils import unpack_nested_arrays


def from_pandas(df: pd.DataFrame, cuda: bool = False, default: t.Tensor = None) -> t.Tensor:
    if df is None and default is None:
        return None

    val = (t.from_numpy(unpack_nested_arrays(df, split_multi_index_rows=False)) if df is not None else default).float()
    return val.cuda() if cuda else val.cpu()


class FittingContext(object):

    class FoldContext(object):
        def __init__(
                self,
                module: t.nn.Module,
                critereon: t.nn.modules.loss._Loss,
                optimzer: t.optim.Optimizer,
        ):
            self.module = module
            self.critereon = critereon
            self.optimzer = optimzer
            self.best_weights = None
            self.best_loss = sys.float_info.max

        def update_best_loss(self, loss):
            if loss < self.best_loss:
                self.best_loss = loss
                self.update_best_weights()

        def update_best_weights(self):
            self.best_weights = deepcopy(self.module.state_dict())

        def restore_best_weights(self):
            self.module.load_state_dict(self.best_weights)

    def __init__(self, override_cv_model: bool):
        self.override_cv_model = override_cv_model
        self.fold_modules: Dict[int, FittingContext.FoldContext] = {}

    def init_if_not_exists(self, fold: int, provider: Callable[[], Tuple]):
        fold = self._translate_fold(fold)
        if fold not in self.fold_modules:
            self.fold_modules[fold] = FittingContext.FoldContext(*provider())

    def update_best_loss(self, fold, loss):
        fold = self._translate_fold(fold)
        self.fold_modules[fold].update_best_loss(loss)

    def restore_best_weights(self):
        for m in self.fold_modules.values():
            m.restore_best_weights()

    def get_module(self, fold):
        fc = self.fold_modules[fold]
        return fc.module, fc.critereon, fc.optimzer

    def _translate_fold(self, fold):
        return 0 if self.override_cv_model else fold