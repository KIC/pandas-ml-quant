from collections import defaultdict

import pandas as pd
import numpy as np


class EarlyStopping(object):

    def __init__(self, patience: int = 0, stop_all_folds: bool = False):
        self.patience = patience
        self.stop_all_folds = stop_all_folds
        self.last_best_losses = defaultdict(lambda: float('inf'))
        self.counts = defaultdict(lambda: 0)
        self.call_conter = 0

    def __call__(self, fold, val_loss):
        self.call_conter += 1
        if val_loss < self.last_best_losses[fold]:
            self.last_best_losses[fold] = val_loss
        else:
            self.counts[fold] += 1
            if self.counts[fold] > self.patience:
                if self.stop_all_folds:
                    raise StopIteration()
                else:
                    raise StopIteration(fold)
