from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
import torch as T
import torch.nn as nn


class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Union[List, Tuple[np.ndarray]], action=None, render_axis=None):
        if isinstance(x, tuple):
            # single sample
            return self.forward([x], [action] if action is not None else None, render_axis)
        else:
            # batch of samples
            state = self._unzip_to_tensor(x)
            if render_axis is not None:
                with T.no_grad():
                    self.render(state, render_axis)

            return self.get_state_value(state) if action is None else self.get_value_for(state, action)

    def _unzip_to_tensor(self, rows):
        # unzip a batch of samples
        if len(rows) < 0:
            # in case of an empty batch return an empty tensor
            return T.FloatTensor([]).to(self.device)

        columns = []
        nr_of_rows = len(rows)
        nr_of_columns = len(rows[0])

        for j in range(nr_of_columns):
            if isinstance(rows[0][j], tuple):
                # if in the first row a column is a tuple we need to unnest these tuples: [([...], [...]), ...]
                list_of_tuples = [row[j] for row in rows]
                columns.append(self._unzip_to_tensor(list_of_tuples))
            else:
                # the values in this column is not a tuple: [[...], ...] so we can stack all of then
                list_of_values = [row[j] for row in rows]
                columns.append(T.FloatTensor(np.vstack(list_of_values)).to(self.device))

        return tuple(columns)

    @property
    @lru_cache(1)
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})

        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))

        return next(iter(devices))

    def get_value_for(self, states: Tuple[List[np.ndarray], ...], action: List) -> T.tensor:
        raise NotImplementedError()

    def get_state_value(self, states: Tuple[List[np.ndarray], ...]) -> Tuple[T.tensor, T.tensor]:
        raise NotImplementedError()

    def render(self, state, ax):
        pass
