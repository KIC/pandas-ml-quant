import torch as t
import logging

PI = t.acos(t.zeros(1)) * 2
_log = logging.getLogger(__name__)


class HeteroscedasticityLoss(t.nn.Module):

    def __init__(self, reduction=None, multi_nominal_reduction='sum'):
        super().__init__()
        self.reduction = reduction
        self.multi_nominal_reduction = multi_nominal_reduction

    def forward(self, y_pred, y_true):
        mu = t.narrow(y_pred, 1, 0, 1)
        sigma = t.exp(t.narrow(y_pred, 1, 1, 1))

        a = 1 / (t.sqrt(2 * PI) * sigma)
        b1 = t.square(mu - y_true)
        b2 = 2 * t.square(sigma)
        b = b1 / b2

        loss = (-t.log(a) + b).squeeze()

        if loss.ndim > 1:
            if self.multi_nominal_reduction == 'sum':
                loss = loss.sum(dim=tuple(range(1, loss.ndim)))
            elif self.multi_nominal_reduction == 'mean':
                loss = loss.mean(dim=tuple(range(1, loss.ndim)))
            elif isinstance(self.multi_nominal_reduction, (tuple, list)):
                raise NotImplementedError("Not yet implemented, sorry :-(")
            else:
                if self.reduction is not None:
                    _log.warning(f"Unknown multi nominal reduction {self.multi_nominal_reduction}, need to be one of ('sum', 'mean', a list of weigts as floats)")

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            if self.reduction is not None:
                _log.warning(f"Unknown reduction {self.reduction}, don't reduce")

            return loss

