from typing import Tuple, List, Generator

import numpy as np

from pandas_ml_utils.ml.data.splitting.sampeling import Sampler


def _get(arr, i):
    return arr[i] if arr is not None else None


class ExtractMultiMultiModelSampler(Sampler):

    def __init__(self, start_index, nr_models, sampler: Sampler):
        super().__init__(sampler.train, sampler.test, None)
        self.__cross_validation = sampler._cross_validation
        self.start_index = start_index
        self.nr_models = nr_models

    def sample(self) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
        cv = self.__cross_validation

        for s in super().sample():
            # features, labels, targets, weights, gross_loss
            (x, y, t, w, gl), (x_val, y_val, t_val, w_val, gl_val) = s
            nr_labels = y.shape[1] // self.nr_models

            def cut(arr):
                return arr[:, (nr_labels * self.start_index):(nr_labels * (self.start_index + 1))]

            _w, _w_val = None, None
            if w is not None:
                if w.ndim > 1:
                    if w.shape[1] > 1:
                        _w = cut(w)
                        _w_val = cut(w_val)
                    else:
                        _w = w[self.start_index]
                        _w_val = w_val[self.start_index]
                else:
                    _w = w
                    _w_val = w_val

            if cv is not None and isinstance(cv, Tuple) and callable(cv[1]):
                for fold_epoch in range(cv[0]):
                    # cross validation, make sure we re-shuffle every fold_epoch
                    y = cut(y)
                    for f, (train_idx, test_idx) in enumerate(cv[1](x, y)):
                        train = (x[train_idx], y[train_idx], _get(t, train_idx), _get(_w, train_idx), _get(gl, train_idx))
                        test = (x[test_idx], y[test_idx], _get(t, test_idx), _get(_w, test_idx), _get(gl, test_idx))
                        yield train, test
            else:
                # fit without cross validation
                yield (x, cut(y), t, _w, gl), (x_val, cut(y_val), t_val, _w_val, gl_val)
