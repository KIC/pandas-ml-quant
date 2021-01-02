import logging
import sys
from typing import Dict

from pandas_ml_utils_torch.pytorch_base import PytochBaseModel

_log = logging.getLogger(__name__)


def take_the_best(folds: Dict[int, PytochBaseModel]):
    best_loss = sys.float_info.max
    best_module = None
    best_fold = -1

    for fold, model in folds.items():
        _log.info(f'fold: {fold}, loss: {model.last_loss}, best loss: {model.best_loss}')
        if model.last_loss < best_loss:
            best_module = model
            best_fold = fold

    _log.info(f'best fold: {best_fold}!')
    return best_module



