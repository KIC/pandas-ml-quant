from keras.layers import Layer, Concatenate
import tensorflow as tf


class Time2Vec(Layer):
    """
    source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                 shape=(self.output_dim,
                                        self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.B = self.add_weight(name='B',
                                 shape=(input_shape[1],
                                        self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.w = self.add_weight(name='w',
                                 shape=(1, 1),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)

        super().build(input_shape)

    def call(self, x, **kwargs):
        from keras import backend as K

        original = self.w * x + self.b
        x = K.repeat_elements(x, self.output_dim, -1)
        sin_trans = K.sin(K.dot(x, self.W) + self.B)
        return K.concatenate([sin_trans ,original], -1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim +1


class tf_Time2Vec(tf.keras.layers.Layer):
    """
    source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    and:    https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                 shape=(self.output_dim,
                                        self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.B = self.add_weight(name='B',
                                 shape=(input_shape[1].value,
                                        self.output_dim),
                                 initializer='uniform',
                                 trainable=True)

        self.w = self.add_weight(name='w',
                                 shape=(1, 1),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1].value, 1),
                                 initializer='uniform',
                                 trainable=True)

        super().build(input_shape)

    def call(self, x, **kwargs):
        K = tf.keras.backend

        original = self.w * x + self.b
        x = K.repeat_elements(x, self.output_dim, -1)
        sin_trans = K.sin(K.dot(x, self.W) + self.B)
        return K.concatenate([sin_trans, original], -1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim +1
