import tensorflow as tf
from keras import backend as _K
from keras.losses import categorical_crossentropy

from pandas_ml_common import LazyInit


def tailed_categorical_crossentropy(nr_of_categories, alpha=0.1, beta=1e10, delta=1, dtype='float32'):
    """
    assuming that we have discretized something like returns where we have less observations in the tails.
    If we want to train a neural net to place returns into the expected bucket we want to penalize if the
    prediction is too close to the mean. we rather want to be pessimistic and force the predictor to
    encounter the tails.

    :param nr_of_categories: number of categories aka length of the one hot encoded vectors
    :param alpha: describes the steepness of the parabola
    :param beta: used for the differentiable_argmax
    :return: returns a keras loss function
    """

    parabolic_penalty = parabolic_crossentropy(nr_of_categories, delta, beta, dtype)

    # custom loss function is cross entropy with penalized tail errors
    def loss_function(y_true, y_pred):
        penalty = alpha * parabolic_penalty(y_pred, y_true)
        loss = categorical_crossentropy(y_true, y_pred)
        return loss + penalty

    return loss_function


def parabolic_crossentropy(nr_of_categories, delta=1, beta=1e10, dtype='float32'):
    argmax = DifferentiableArgmax(nr_of_categories, beta, dtype=dtype)
    offset = LazyInit(lambda: _K.constant(delta / 2, dtype=dtype))
    f = LazyInit(lambda: _K.constant((nr_of_categories + delta) / nr_of_categories, dtype=dtype))

    def loss_function(y_true, y_pred):
        return ((argmax(y_pred) + offset()) - (argmax(y_true)) * f()) ** 2

    return loss_function


def normal_penalized_crossentropy(nr_of_categories, alpha=10, beta=1e10, max_z=4, dtype='float32'):
    max_cat_int = nr_of_categories - 1
    _alpha = LazyInit(lambda: _K.constant(alpha, dtype=dtype))
    _mue = LazyInit(lambda: _K.constant(max_cat_int / 2, dtype=dtype))
    _sigma = LazyInit(lambda: _K.constant((max_cat_int - max_cat_int / 2) / abs(max_z), dtype=dtype))
    _argmax = DifferentiableArgmax(nr_of_categories, beta, dtype=dtype)
    _norm_dist = NormDistWeightedPenalizer(dtype)

    def loss_function(y_true, y_pred):
        # 1st we calculate the cross entropy
        loss = categorical_crossentropy(y_true, y_pred)

        # then we use the normal shaped probability distrubution to penalize the prediction
        penalty = _norm_dist(_argmax(y_pred), _mue(), _sigma()) * _alpha()

        return loss * penalty

    return loss_function


class DifferentiableArgmax(object):

    def __init__(self, nr_of_categories, beta=1e10, dtype='float32'):
        self.y_range = LazyInit(lambda: _K.arange(0, nr_of_categories, dtype=dtype))
        self.beta = beta

    def __call__(self, *args, **kwargs):
        y = args[0]
        return _K.sum(_K.softmax(y * self.beta) * self.y_range(), axis=-1)


class NormDistWeightedPenalizer(object):

    def __init__(self, dtype='float32'):
        self.one = LazyInit(lambda: _K.constant(1, dtype=dtype))
        self.two = LazyInit(lambda: _K.constant(2, dtype=dtype))
        self.pi = LazyInit(lambda: _K.constant(3.1415, dtype=dtype))
        self.e = LazyInit(lambda: _K.constant(2.7182, dtype=dtype))

    def __call__(self, *args, **kwargs):
        x, mu, sigma = args
        one, two, pi, e = self.one(), self.two(), self.pi(), self.e()

        # 1 / (sqrt(2pi) * sigma) * e ** -((x - mu)**2 / 2sigma**2)
        return  (one / (_K.sqrt(two * pi) * sigma)) * e ** -((x - mu) ** two / two * sigma ** two)


def mdn_cost(y_t, y_p):
    mu = y_p[:, 0]
    sigma = y_p[:, 1]

    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y_t))
