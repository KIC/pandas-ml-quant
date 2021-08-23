from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from typing import List, Callable, Type, Dict

import pandas as pd
import torch as t
import torch.nn as nn

from pandas_ml_common import Typing
from pandas_ml_common.utils import to_pandas
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.model.base_model import Model, AutoEncoderModel
from pandas_ml_utils.ml.summary import Summary
from pandas_ml_utils_torch.pytorch_base import PytorchNN, PytochBaseModel
from pandas_ml_utils_torch.utils import from_pandas

_log = logging.getLogger(__name__)


class _AbstractPytorchModel(Model):

    def __init__(self,
                 module_provider: Type[PytorchNN],
                 features_and_labels: FeaturesAndLabels,
                 criterion_provider: Type[nn.modules.loss._Loss],
                 optimizer_provider: Type[t.optim.Optimizer],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[Typing.PatchedDataFrame], Forecast] = None,
                 restore_best_weights: bool = False,
                 merge_cross_folds: Callable[[Dict[int, PytochBaseModel]], PytochBaseModel] = None,
                 callbacks: Dict[str, List[Callable]] = {},
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, forecast_provider, **kwargs)
        self.merge_cross_folds = merge_cross_folds
        self.callbacks = callbacks

        self.log_once = LogOnce().log
        self._current_model: PytochBaseModel = PytochBaseModel(module_provider, criterion_provider, optimizer_provider, restore_best_weights, False, **kwargs)
        self._models: Dict[int, PytochBaseModel] = defaultdict(self._current_model.shadow_copy)
        self._cuda = False

    def init_fit(self, **kwargs):
        self._cuda = kwargs.get("cuda", False)

    def init_fold(self, epoch: int, fold: int):
        self._current_model = self._models[fold]

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, weight: pd.DataFrame, fold: int, **kwargs):
        self._current_model.fit_epoch(from_pandas(x, self._cuda), from_pandas(y, self._cuda), from_pandas(weight, self._cuda))

    def calculate_loss(self, fold, x, y_true, weight) -> float:
        # return self._current_model.calculate_loss(
        return self._models[fold].calculate_loss(
            from_pandas(x, self._cuda),
            from_pandas(y_true, self._cuda),
            from_pandas(weight, self._cuda),
        )

    def merge_folds(self, epoch: int):
        if len(self._models) > 1:
            if self.merge_cross_folds is not None:
                self._current_model = self.merge_cross_folds(self._models)
                self._models.clear()
            else:
                self.log_once("merge_folds", _log.warning,
                              f"no merge folds function supplied, keep training {len(self._models)} independent folds")
        else:
            self._current_model = deepcopy(self._models[0])

    def finish_learning(self):
        if len(self._models) > 1 and self.merge_cross_folds is not None:
            # this is bad as our current_model never got updated and is just a randomly initialized model
            # we want at least warn the user about this
            _log.warning("cross foling was used whithout a merging strategy, the main model stayes untrained!!!")

        self._models.clear()

    @property
    def module(self):
        return self._current_model if self._current_model is None else self._current_model.net

    def _predict(self, features: pd.DataFrame, col_names, samples=1, cuda=False, **kwargs) -> Typing.PatchedDataFrame:
        if cuda:
            self._current_model = self._current_model.cuda()
        else:
            self._current_model = self._current_model.cpu()

        return to_pandas(self._current_model.predict(from_pandas(features, cuda), samples), features.index, col_names)


class PytorchModel(_AbstractPytorchModel):

    @wraps(_AbstractPytorchModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, cuda=False, **kwargs) -> Typing.PatchedDataFrame:
        self._current_model.eval()
        return self._predict(features, self._labels_columns, samples, cuda, **kwargs)


class PytorchAutoEncoderModel(_AbstractPytorchModel, AutoEncoderModel):

    @wraps(_AbstractPytorchModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _auto_encode(self, features: pd.DataFrame, samples, cuda=False, **kwargs) -> Typing.PatchedDataFrame:
        self._current_model.eval()
        return self._predict(features, self._labels_columns, samples, cuda, **kwargs)

    def _encode(self, features: pd.DataFrame, samples, cuda=False, **kwargs) -> Typing.PatchedDataFrame:
        self._current_model.eval()
        return self._predict(features, self._features_and_labels.latent_names, samples, cuda, **kwargs)

    def _decode(self, latent_features: pd.DataFrame, samples, cuda=False, **kwargs) -> Typing.PatchedDataFrame:
        self._current_model.eval()
        return self._predict(latent_features,  self._labels_columns, samples, cuda, **kwargs)

