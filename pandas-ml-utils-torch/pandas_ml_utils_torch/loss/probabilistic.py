import torch as t

PI = t.acos(t.zeros(1)) * 2


class HeteroscedasticityLoss(t.nn.Module):

    def __init__(self, reduction=None):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        mu = t.narrow(y_pred, 1, 0, 1)
        sigma = t.exp(t.narrow(y_pred, 1, 1, 1))

        a = 1 / (t.sqrt(2 * PI) * sigma)
        b1 = t.square(mu - y_true)
        b2 = 2 * t.square(sigma)
        b = b1 / b2

        loss = (-t.log(a) + b).squeeze()
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

