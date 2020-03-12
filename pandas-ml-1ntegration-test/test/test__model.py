from unittest import TestCase

from sklearn.neural_network import MLPClassifier

import pandas_ml_quant
from pandas_ml_utils import FeaturesAndLabels, SkModel
from test.config import DF_TEST

print(pandas_ml_quant.__version__)


class TestModel(TestCase):

    def test_simple_model(self):
        df = DF_TEST.copy()

        # TODO
        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].q.ta_rsi().q.ta_shape_for_auto_regression(280),
                        lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_shape_for_auto_regression(280)
                    ],
                    labels=[
                        lambda df: df["Close"] > df["Open"],
                    ],
                    min_required_samples=280
                ),
                # kwargs
                forecasting_time_steps=7
            )
        )

        print(fit)

    def test_hyper_parameter_for_simple_model(self):

        pass

