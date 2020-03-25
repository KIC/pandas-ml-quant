import os
from unittest import TestCase

import numpy as np
from keras import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from sklearn.neural_network import MLPClassifier, MLPRegressor

import pandas_ml_quant
from pandas_ml_utils import FeaturesAndLabels, SkModel, KerasModel
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor
from pandas_ml_utils.ml.data.splitting import RandomSplits
from pandas_ml_utils.ml.data.splitting.sampeling import KFoldBoostRareEvents, KEquallyWeightEvents
from pandas_ml_utils.ml.summary import ClassificationSummary, RegressionSummary
from test.config import DF_TEST

print(pandas_ml_quant.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestModel(TestCase):

    def test_simple_regression_model(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPRegressor(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].q.ta_rsi().q.ta_rnn(28),
                        lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_rnn(28)
                    ],
                    labels=[
                        lambda df: (df["Close"] / df["Open"] - 1).shift(-1),
                    ]
                ),
                summary_provider=RegressionSummary
            )
        )

        print(fit)
        html = fit._repr_html_()

        prediction = df.model.predict(fit.model)
        print(prediction)
        self.assertIsInstance(prediction[PREDICTION_COLUMN_NAME, 0].iloc[-1], (float, np.float, np.float32, np.float64))

        backtest = df.model.backtest(fit.model)


    def test_simple_classification_model(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].q.ta_rsi().q.ta_rnn(28),
                        lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_rnn(28)
                    ],
                    labels=[
                        lambda df: (df["Close"] > df["Open"]).shift(-1),
                    ]
                ),
                # kwargs
                forecasting_time_steps=7
            )
        )

        print(fit)
        html = fit._repr_html_()

        prediction = df.model.predict(fit.model)
        print(prediction)
        self.assertIsInstance(prediction[PREDICTION_COLUMN_NAME, 0].iloc[-1], (float, np.float, np.float32, np.float64))

        backtest = df.model.backtest(fit.model)

        # test multiple samples
        samples = df.model.predict(fit.model, samples=2)
        self.assertIsInstance(samples[PREDICTION_COLUMN_NAME, 0].iloc[-1], list)
        self.assertEqual(2, len(samples[PREDICTION_COLUMN_NAME, 0].iloc[-1]))

    def test_keras_model(self):
        df = DF_TEST.copy()

        def model_provider():
            model = Sequential([
                Reshape((28 * 2,), input_shape=(28, 2)),
                Dense(60, activation='tanh'),
                Dense(50, activation='tanh'),
                Dense(1, activation="sigmoid")
            ])

            model.compile(Adam(), loss='mse')

            return model

        fit = df.model.fit(
            KerasModel(
                model_provider,
                FeaturesAndLabels(
                    features=extract_with_post_processor([
                        lambda df: df["Close"].q.ta_rsi(),
                        lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).rename("RelVolume")
                    ], lambda df: df.q.ta_rnn(28)),
                    labels=[
                        lambda df: (df["Close"] > df["Open"]).shift(-1),
                    ]
                ),
                # kwargs
                forecasting_time_steps=7,
                epochs=2
            )
        )

        print(fit)

        prediction = df.model.predict(fit.model)
        print(prediction)
        print(type(prediction[PREDICTION_COLUMN_NAME, 0].iloc[-1]))
        self.assertIsInstance(prediction[PREDICTION_COLUMN_NAME, 0].iloc[-1], (float, np.float, np.float32, np.float64))

        backtest = df.model.backtest(fit.model)

    # FIXME implement functionality such that test passes
    def _test_hyper_parameter_for_simple_model(self):
        from hyperopt import hp

        """given"""
        df = DF_TEST.copy()
        df['label'] = df["spy_Close"] > df["spy_Open"]

        """when fit with find hyper parameter"""
        fit = df.fit(
            SkModel(MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42),
                    FeaturesAndLabels(features=['vix_Close'], labels=['label'],
                                      target_columns=["vix_Open"],
                                      loss_column="spy_Volume")),
            test_size=0.4,
            test_validate_split_seed=42,
            hyper_parameter_space={'alpha': hp.choice('alpha', [0.0001, 10]), 'early_stopping': True, 'max_iter': 50,
                                   '__max_evals': 4, '__rstate': np.random.RandomState(42)}
        )

        """then test best parameter"""
        self.assertEqual(fit.model.skit_model.get_params()['alpha'], 0.0001)


        pass

    def test_KFold(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df["Close"].q.ta_trix(),
                            lambda df: df["Close"].q.ta_ppo(),
                            lambda df: df["Close"].q.ta_apo(),
                            lambda df: df["Close"].q.ta_macd(),
                            lambda df: df.q.ta_adx(),
                        ],
                        lambda df: df.q.ta_rnn(range(10))
                    ),
                    labels=[
                        lambda df: df["Close"].q.ta_sma(period=60) \
                            .q.ta_cross(df["Close"].q.ta_sma(period=20)) \
                            .q.ta_rnn([1, 2, 3, 4, 5]) \
                            .abs() \
                            .sum(axis=1) \
                            .shift(-5) \
                            .astype(bool)

                    ]
                )
            ),
            RandomSplits(test_size=0.4,
                         test_validate_split_seed=42,
                         cross_validation=(1, KFoldBoostRareEvents(n_splits=5).split))
        )

    def test_debug(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, warm_start=True, max_iter=2),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df["Close"].q.ta_macd().ml[['macd.*', 'signal.*']],
                            lambda df: df.q.ta_adx().ml[['+DI', '-DM', '+DM']],
                            lambda df: df["Close"].q.ta_mom(),
                            lambda df: df["Close"].q.ta_apo(),
                            lambda df: df.q.ta_atr(),
                            lambda df: df["Close"].q.ta_trix(),
                        ],
                        lambda df: df.q.ta_rnn(280)
                    ),
                    labels=[
                        lambda df: df["Close"].q.ta_future_bband_quantile().q.ta_one_hot_encode_discrete()
                    ],
                    targets=[
                        lambda df: df["Close"].q.ta_bbands()[["lower", "upper"]]
                    ]
                ),
                summary_provider=ClassificationSummary,
            ),
            RandomSplits(test_size=0.4,
                         test_validate_split_seed=42,
                         cross_validation=(1, KEquallyWeightEvents(n_splits=3).split))
        )

        print(fit)
        prediction = df.model.predict(fit.model, tail=3)
        self.assertEqual(3, len(prediction))

        target_predictions = prediction.map_prediction_to_target()
        print(target_predictions)
        self.assertEqual(9, len(target_predictions))

