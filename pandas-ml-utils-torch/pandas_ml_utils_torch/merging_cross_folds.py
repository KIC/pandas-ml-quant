import logging
import sys
from typing import Dict

from pandas_ml_utils_torch.utils import FittingContext

_log = logging.getLogger(__name__)


def take_the_best(folds: Dict[int, FittingContext.FoldContext]):
    best_loss = sys.float_info.max
    best_module = None
    best_fold = -1

    for fold, ctx in folds.items():
        _log.info(f'fold: {fold}, loss: {ctx.last_loss}')
        if ctx.last_loss < best_loss:
            best_module = ctx.module
            best_fold = fold

    _log.info(f'best fold: {best_fold}!')
    return best_module.state_dict()



