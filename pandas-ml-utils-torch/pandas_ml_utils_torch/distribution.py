from torch.distributions import Uniform, SigmoidTransform, AffineTransform, TransformedDistribution
from torch.distributions import constraints
import torch as t


class Logistic(TransformedDistribution):
    r"""
    Creates a logistic distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Uniform` distribution transformed with the
    `SigmoidTransform` and `AffineTransform` such that:
        X ~ Logistic(loc, scale)
        cdf(x; loc, scale) = 1 / (1 + exp(-(x - loc) / scale))
    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution
    Example::
        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)
        >>> # of the base Normal distribution
        >>> m = distributions.Logistic(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 1.2082, -3.3503,  0.0038])
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Uniform(t.zeros(loc.shape), t.ones(loc.shape), validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super(Logistic, self).__init__(
            base_dist,
            [SigmoidTransform().inv, AffineTransform(loc, scale)]
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Logistic, _instance)
        return super(Logistic, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.transforms[-1].loc

    @property
    def scale(self):
        return self.transforms[-1].scale
