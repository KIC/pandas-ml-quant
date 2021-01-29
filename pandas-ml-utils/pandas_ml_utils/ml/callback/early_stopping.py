from collections import defaultdict


class EarlyStopping(object):

    def __init__(self, patience: int = 0, tolerance: float = 0, stop_all_folds: bool = True):
        self.patience = patience
        self.tolerance = tolerance
        self.stop_all_folds = stop_all_folds
        self.last_best_losses = defaultdict(lambda: float('inf'))
        self.counts = defaultdict(lambda: 0)
        self.call_counter = 0

    def __call__(self, fold, val_loss):
        self.call_counter += 1
        if val_loss - self.last_best_losses[fold] < self.tolerance:
            self.last_best_losses[fold] = val_loss
        else:
            self.counts[fold] += 1
            if self.counts[fold] > self.patience:
                if self.stop_all_folds:
                    raise StopIteration()
                else:
                    raise StopIteration(fold)
