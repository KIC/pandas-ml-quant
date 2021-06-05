from unittest import TestCase

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import layers as tfl
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras import Sequential


from pandas_ml_common import naive_splitter
from pandas_ml_utils import FeaturesAndLabels, FittingParameter
from pandas_ml_utils_tf.tf_model import TFModel
from pandas_ml_utils_tf.tf_nn import TensorflowNN, TensorflowNNFactory


class TestNN(TestCase):
    #libcusolver.so
    #'libcusparse.so.11';
    #'libcudnn.so.8

    def test_scratch2(self):

        # create a batch of images
        x = np.random.normal(5, 0.2, (10, 1))

        factory = TensorflowNNFactory.create(
            Sequential([InputLayer(input_shape=(1,)), Dense(2, activation='relu')]),
            lambda net, x: net(x),
            lambda net, x: tf.stack([net(x)[...,0], tf.exp(net(x)[...,1])], axis=-1)
        )

        nn = factory()
        print("train", tf.function(nn.forward_training)(tf.convert_to_tensor( x )))
        print("predict", tf.function(nn.forward_predict)(tf.convert_to_tensor( x )))
        print(nn.trainable_variables())

    def test_scratch(self):
        input_shape = [28]
        encoded_shape = 2

        # create a batch of images
        x = np.random.normal(5, 0.2, (10, *input_shape))

        class NN(TensorflowNN):

            def __init__(self) -> None:
                super().__init__()
                self.dist = tfl.IndependentNormal(encoded_shape)

                self.model = Sequential([
                    InputLayer(input_shape=input_shape),
                    Dense(10, activation='relu'),
                    Dense(tfl.IndependentNormal.params_size(encoded_shape))
                ])

            def trainable_variables(self):
                return self.model.trainable_variables + [self.dist]

            def forward_training(self, *input) -> tf.Tensor:
                return self.dist(self.model(*input))

            def forward_predict(self, *input) -> tf.Tensor:
                return self.model(*input)  # add exp


        nn = NN()
        print(tf.function(nn.forward_training)(tf.convert_to_tensor( x )))
        print(tf.function(nn.forward_predict)(tf.convert_to_tensor( x )))
        print(nn.trainable_variables())


        nn2 = NN()
        print(tf.function(nn2.forward_training)(tf.convert_to_tensor( x )))
        print(tf.function(nn2.forward_predict)(tf.convert_to_tensor( x )))
        print(nn2.trainable_variables())

        @tf.function
        def nll(y, y_pred):
            y_pred = tfd.Normal(loc=y_pred[..., 0], scale=y_pred[..., 1])
            print(y_pred)
            return -tf.reduce_mean(y_pred.log_prob(y))

        df = pd.DataFrame({"x": [_x.tolist() for _x in x]})
        tfm = TFModel(
            NN,
            FeaturesAndLabels("x", "x"),
            nll, #tf.keras.losses.mse,
            tf.keras.optimizers.Adam(learning_rate=1e-1)
        )

        tfm.init_fit()
        tfm.init_fold(0, 0)
        tfm.fit_batch(df, df, None, 0)


    def test_nn(self):
        """
        N = 1000
        df = pd.DataFrame({
            "x": np.random.normal(5, 0.2, N)
        })

        class NN(TensorflowNN):

            def __init__(self, *args, **kwargs):
                # Making a distribution using the flow and a N(0,1)
                tfl.Dense(tfl.IndependentNormal.params_size(encoded_shape)),
                tfl.IndependentNormal(encoded_shape)

            self.b = tf.Variable(0.0)
                self.a = tf.Variable(1.0)
                bijector = tfb.AffineScalar(shift=self.b, scale=self.a)
                self.dist = tfd.TransformedDistribution(distribution=tfd.Normal(loc=0, scale=1), bijector=bijector)

            def forward_training(self, *input) -> tf.Tensor:
                return -self.dist.log_prob(input[0])

            def forward_predict(self, *input) -> tf.Tensor:
                return tf.concat([self.a, self.b], axis=0)

        net = NN()
        print(net.forward_training(tfd.Normal(loc=5, scale=0.2).sample(1000)).numpy())
        print(net.forward_predict(None).numpy())

        opt = tf.keras.optimizers.Adam(learning_rate=1e-1)
        with tf.Graph().as_default():
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(net.forward_training(tfd.Normal(loc=5, scale=0.2).sample(1000)))
                gradients = tape.gradient(loss, net)  # how to get net.trainable_variables
                opt.apply_gradients(zip(gradients, net))

        fit = df.model.fit(
            TFModel(
                NN,
                FeaturesAndLabels("x", "x"),
                lambda y_real, y: 0, # -tf.reduce_mean(dist.log_prob(y)),
                tf.keras.optimizers.Adam(learning_rate=1e-1)
            ),
            FittingParameter(epochs=1000, splitter=naive_splitter(0.5))
        )

        print(fit)
        """
