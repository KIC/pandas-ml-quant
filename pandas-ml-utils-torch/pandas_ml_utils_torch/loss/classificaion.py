import torch as t
import torch.nn as nn

from pandas_ml_utils_torch.loss import CrossEntropyLoss
from ._loss_utils import reduce


class DifferentiableArgmax(nn.Module):

    def __init__(self, nr_of_categories, beta=1e10):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('y_range', t.arange(0, nr_of_categories).float())
        self.register_buffer('beta', t.tensor(beta).float())

    def forward(self, y):
        return t.sum(self.softmax(y * self.beta) * self.y_range, dim=-1)


class ParabolicPenaltyLoss(nn.Module):

    def __init__(self, nr_of_categories, delta=1.0, beta=1e10):
        super().__init__()
        self.argmax = DifferentiableArgmax(nr_of_categories, beta)
        self.register_buffer('offset', t.tensor(delta / 2).float())
        self.register_buffer('f', t.tensor((nr_of_categories + delta) / nr_of_categories).float())

    def forward(self, y_pred, y_true):
        return ((self.argmax(y_pred) + self.offset) - (self.argmax(y_true)) * self.f) ** 2


class TailedCategoricalCrossentropyLoss(nn.Module):

    def __init__(self, nr_of_categories: int, alpha=0.1, beta=1e10, delta=1.0, reduction='none'):
        """
        assuming that we have discretized something like returns where we have less observations in the tails.
        If we want to train a neural net to place returns into the expected bucket we want to penalize if the
        prediction is too close to the mean. we rather want to be pessimistic and force the predictor to
        encounter the tails.

        :param nr_of_categories: number of categories aka length of the one hot encoded vectors
        :param alpha: describes the steepness of the parabola
        :param beta: used for the differentiable_argmax
        :param delta: is used to un-evenly skew the loss to the outer bounds. 0 now skew > bigger skew
        :return: returns a keras loss function
        """
        super().__init__()
        self.parabolic_penalty = ParabolicPenaltyLoss(nr_of_categories, delta, beta)
        self.categorical_crossentropy = CrossEntropyLoss()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_pred.shape)
        penalty = self.alpha * self.parabolic_penalty(y_true, y_pred).squeeze()
        loss = self.categorical_crossentropy(y_pred, y_true)
        loss = penalty + loss

        return reduce(loss, self.reduction, loss.view(y_pred.shape[0]))
