import torch as t
import torch.nn as nn


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
