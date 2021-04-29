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


def average_folds(folds: Dict[int, PytochBaseModel]):
    mean_model = None

    # add all parameters
    for i, (fold, model) in enumerate(folds.items()):
        if i == 0:
            mean_model = model.shadow_copy()
        else:
            for target_param, local_param in zip(mean_model.net.parameters(), model.net.parameters()):
                target_param.data.copy_(local_param.data + target_param.data)

    # average all parameters
    for target_param in mean_model.net.parameters():
        target_param.data.copy_(target_param.data / len(folds))

    return mean_model


