import torch.nn as nn


class Reshape(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        # keep batch size
        return x.view(x.shape[0], *self.shape)
