from typing import Union, Tuple, Callable

import torch as t
import torch.nn as nn
from torch.nn import init


class LambdaSplitter(nn.Module):

    def __init__(self,  *funcs: Callable[[t.Tensor], t.Tensor]) -> None:
        super().__init__()
        self.funcs = funcs

    def forward(self, input, *args):
        return tuple([c(input) for c in self.funcs])


class Flatten(nn.Module):

    def __init__(self, *args):
        super().__init__()

    def forward(self, input, *args):
        return input.view(input.size(0), -1)


class Squeeze(nn.Module):

    def __init__(self, *args):
        super().__init__()

    def forward(self, input, *args):
        squoze = input.squeeze()

        # Note that we need to keep the batch dimension in case it is 1
        return squoze if input.shape[0] > 1 else squoze.view(1, *squoze.shape)


class Reshape(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        # keep batch size
        return x.view(x.shape[0], *self.shape)


class KerasLikeLSTM(nn.Module):

    def __init__(self, input_shape, output, return_sequence=False, sequential=False):
        super().__init__()
        self.features = input_shape[1] if isinstance(input_shape, tuple) else 1
        self.timesteps = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        self.output = output
        self.lstm = nn.LSTM(self.features, self.output, batch_first=True)
        self.return_sequence = return_sequence
        self.sequential = sequential
        self.hidden = None
        self.cell = None

    def forward(self, x):
        if not self.sequential or self.hidden is None:
            self.hidden = t.zeros(1, x.size(0), self.output).to(next(self.parameters()).device)

        self.cell = t.zeros(1, x.size(0), self.output).to(next(self.parameters()).device)
        out, self.hidden = self.lstm(x, (self.hidden, self.cell))

        if self.return_sequence:
            return out
        else:
            # only return last dimension
            return out[:, -1]


class ResidualLayer(nn.Module):

    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.l2 = nn.Linear(output_size, output_size)
        self.activation = activation

    def forward(self, x):
        x1 = self.activation(self.l1(x))
        x2 = self.activation(self.l2(x1))
        return x1 + x2


class Time2Vec(nn.Module):
    """
    source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, in_dim: Union[Tuple[int, int], int], out_dim: int):
        super().__init__()
        if not isinstance(in_dim, tuple):
            in_dim = (in_dim, 1)

        self.output_dim = out_dim
        self.w = nn.Parameter(t.Tensor(1, 1))
        self.b = nn.Parameter(t.Tensor(in_dim[0], 1))
        self.W = nn.Parameter(t.Tensor(out_dim * in_dim[1], out_dim * in_dim[1]))
        self.B = nn.Parameter(t.Tensor(in_dim[0], out_dim * in_dim[1]))

    def reset_parameters(self):
        init.uniform_(self.W, 1e-6, 1)
        init.uniform_(self.B, 1e-6, 1)
        init.uniform_(self.w, 1e-6, 1)
        init.uniform_(self.b, 1e-6, 1)

    def forward(self, x):
        if x.ndim < 3:
            x = x.view(*x.shape, 1)

        original = self.w * x + self.b
        x = t.repeat_interleave(x, self.output_dim, dim=-1)
        sin_trans = t.sin((x @ self.W) + self.B)
        return t.cat([sin_trans, original], -1)

