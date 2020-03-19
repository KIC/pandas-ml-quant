from collections import Callable

import tensorflow as tf
from keras import backend as K, Input
from keras.initializers import Constant
from keras.layers import Layer, Dense, Concatenate


class NormalDistributionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        mu = Dense(1)(inputs)
        sigma = Dense(1, activation=lambda x: tf.nn.elu(x) + 1)(inputs)
        out = Concatenate()([mu, sigma])
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], 2


class CurveFit(Layer):

    def __init__(self, parameters: int, function: Callable, initializer='uniform', **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        self.function = function
        self.initializer = initializer
        self.batch_size = None
        self.kernel = None

    def build(self, input_shape):
        """
        parameters: pass the number of parameters of the function you try to fit
        function: pass the function you want to fit i.e. `lambda x, args: x * sum(args)`
        """

        # set batch size
        self.batch_size = input_shape[0]

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.parameters,),
                                      initializer=self.initializer,
                                      trainable=True)

        # Be sure to call this at the end
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # here we gonna invoke out function and return the result.
        # the loss function will do whatever is needed to fit this function as good as possible
        return self.function(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


class LinearRegressionLayer(CurveFit):

    def __init__(self):
        super().__init__(2, LinearRegressionLayer.fit)

    @staticmethod
    def fit(x, args):
        # y = k * x + d
        return args[0] * K.arange(0, x.shape[1], 1, dtype=x.dtype) + args[1]


class LPPLLayer(CurveFit):
    # original model:
    #  dt = tc - t
    #  dtPm = dt ^ m
    #  A + B * dtPm + C * dtPm * cos(w * ln(dt) - phi)
    def __init__(self):
        super().__init__(3, self.lppl, Constant(0.5))

    def get_tc(self):
        return self.get_weights()[0][0]

    def lppl(self, x, args):
        N = K.constant(int(x.shape[-1]), dtype=x.dtype)
        t = K.arange(0, int(x.shape[-1]), 1, dtype=x.dtype)

        # note that we need to get the variables to be centered around 0
        # so to correct the magnitude we offset them by constants
        # w just has a magnitude of 10s from empirical results
        # for tc we apply a factor of 5 which should be interpreted as week
        # a tc 0.5 means half a week in the future
        tc = args[0] * K.constant(5, dtype=x.dtype) + N
        m = args[1]
        w = args[2] * K.constant(10, dtype=x.dtype)

        # then we calculate the lppl with the given parameters
        dt = (tc - t)
        dtPm = K.pow(dt, m)
        dtln = K.log(dt)
        abcc = LPPLLayer.matrix_equation(x, dtPm, dtln, w, N)
        a, b, c1, c2 = (abcc[0], abcc[1], abcc[2], abcc[3])

        # LPPL = A+B(tc −t)^m +C1(tc −t)^m cos(ω ln(tc −t)) +C2(tc −t)^m sin(ω ln(tc −t))
        return a + b * dtPm + c1 * dtPm * K.cos(w * dtln) + c2 * dtPm * K.sin(w * dtln)

    @staticmethod
    def matrix_equation(x, dtPm, dtln, w, N):
        fi = dtPm
        gi = dtPm * K.cos(w * dtln)
        hi = dtPm * K.sin(w * dtln)

        fi_pow_2 = K.sum(fi * fi)
        gi_pow_2 = K.sum(gi * gi)
        hi_pow_2 = K.sum(hi * hi)

        figi = K.sum(fi * gi)
        fihi = K.sum(fi * hi)
        gihi = K.sum(gi * hi)

        # note that our price is already a log price so we should not log it one more time
        yi = x  # K.log(x)
        yifi = K.sum(yi * fi)
        yigi = K.sum(yi * gi)
        yihi = K.sum(yi * hi)

        fi = K.sum(fi)
        gi = K.sum(gi)
        hi = K.sum(hi)
        yi = K.sum(yi)

        A = K.stack([
            K.stack([N, fi, gi, hi]),
            K.stack([fi, fi_pow_2, figi, fihi]),
            K.stack([gi, figi, gi_pow_2, gihi]),
            K.stack([hi, fihi, gihi, hi_pow_2])
        ], axis=0)

        b = K.stack([yi, yifi, yigi, yihi])

        # do a classic x = (A'A)⁻¹A' b
        return tf.linalg.solve(A, K.reshape(b, (4, 1)))
