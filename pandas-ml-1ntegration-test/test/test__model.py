import os
from unittest import TestCase

import numpy as np
from keras import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

import pandas_ml_quant
from pandas_ml_quant_rl.model.rl_trading_agent import TradingAgentGym
from pandas_ml_utils import FeaturesAndLabels, SkModel, KerasModel, ReinforcementModel, Constant
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor
from pandas_ml_utils.ml.data.splitting import RandomSplits, RandomSequences, NaiveSplitter
from pandas_ml_utils.ml.data.splitting.sampeling import KFoldBoostRareEvents, KEquallyWeightEvents
from pandas_ml_utils.ml.summary import ClassificationSummary, RegressionSummary
from test.config import DF_TEST

print(pandas_ml_quant.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestModel(TestCase):

    def test_linear_model(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                Lasso(),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].ta.rsi().ta.rnn(28),
                        lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(28)
                    ],
                    labels=[
                        lambda df: (df["Close"] / df["Open"] - 1).shift(-1),
                    ]
                ),
                summary_provider=RegressionSummary
            ),
            NaiveSplitter()
        )

        print(fit)

        prediction = df.model.predict(fit.model)
        print(prediction)

        backtest = df.model.backtest(fit.model)

    def test_simple_regression_model(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPRegressor(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, max_iter=2),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["Close"].ta.rsi().ta.rnn(28),
                        lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(28)
                    ],
                    labels=[
                        lambda df: (df["Close"] / df["Open"] - 1).shift(-1),
                    ]
                ),
                summary_provider=RegressionSummary
            ),
            NaiveSplitter()
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
                        lambda df: df["Close"].ta.rsi().ta.rnn(28),
                        lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(28)
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
                        lambda df: df["Close"].ta.rsi(),
                        lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).rename("RelVolume")
                    ], lambda df: df.ta.rnn(28)),
                    labels=[
                        lambda df: (df["Close"] > df["Open"]).shift(-1),
                    ],
                    sample_weights=["Volume"]
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
                            lambda df: df["Close"].ta.trix(),
                            lambda df: df["Close"].ta.ppo(),
                            lambda df: df["Close"].ta.apo(),
                            lambda df: df["Close"].ta.macd(),
                            lambda df: df.ta.adx(),
                        ],
                        lambda df: df.ta.rnn(range(10))
                    ),
                    labels=[
                        lambda df: df["Close"].ta.sma(period=60) \
                            .ta.cross(df["Close"].ta.sma(period=20)) \
                            .ta.rnn([1, 2, 3, 4, 5]) \
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

    def test_future_bband_quantile_clasification(self):
        df = DF_TEST.copy()

        fit = df.model.fit(
            SkModel(
                MLPClassifier(activation='tanh', hidden_layer_sizes=(60, 50), random_state=42, warm_start=True, max_iter=2),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df["Close"].ta.macd()._[['macd.*', 'signal.*']],
                            lambda df: df.ta.adx()._[['+DI', '-DM', '+DM']],
                            lambda df: df["Close"].ta.mom(),
                            lambda df: df["Close"].ta.apo(),
                            lambda df: df.ta.atr(),
                            lambda df: df["Close"].ta.trix(),
                        ],
                        lambda df: df.ta.rnn(280)
                    ),
                    labels=[
                        lambda df: df["Close"].ta.future_bband_quantile(include_mean=False).ta.one_hot_encode_discrete()
                    ],
                    targets=[
                        lambda df: df["Close"].ta.bbands()[["lower", "upper"]]
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
        self.assertEqual((3,), np.array(prediction[PREDICTION_COLUMN_NAME].iloc[-1, -1]).shape)

        target_predictions = prediction.map_prediction_to_target()
        print(target_predictions)
        self.assertEqual(9, len(target_predictions))

    def test_reinformcement(self):
        # given a data frame
        df = DF_TEST.copy()

        # and a trading agent
        class ARGym(TradingAgentGym):

            def calculate_trade_reward(self, portfolio_performance_log):
                return portfolio_performance_log["net"].iloc[-1]

            def next_observation(self, idx, features, labels, targets, weights=None, gross_loss=None):
                return features

        # when we fit the agent
        fit = df.model.fit(
            ReinforcementModel(
                lambda: PPO2('MlpLstmPolicy',
                             DummyVecEnv([lambda: ARGym((28, 2), initial_capital=100000)]),
                             nminibatches=1),
                FeaturesAndLabels(
                    features=extract_with_post_processor(
                        [
                            lambda df: df.ta.atr(),
                            lambda df: df["Close"].ta.trix(),
                        ],
                        lambda df: df.ta.rnn(28)
                    ),
                    targets=[
                        lambda df: df["Close"]
                    ],
                    labels=[Constant(0)],
                )
            ),
            RandomSequences(0.1, 0.7, max_folds=None),
            total_timesteps=128 * 2,
            verbose=1,
            render='system'
        )

        print(fit.test_summary.df[PREDICTION_COLUMN_NAME])

        prediction = df.model.predict(fit.model, tail=3)
        print(prediction[PREDICTION_COLUMN_NAME])
        self.assertEqual(3, len(prediction))
        self.assertGreater(len(fit.model.reward_history), 0)
        self.assertGreater(len(fit.model.reward_history[0]), 1)
        self.assertGreater(len(fit.model.reward_history[0][1]), 1)
        backtest = df.model.backtest(fit.model).df
        print(backtest[PREDICTION_COLUMN_NAME])
        self.assertEqual(3, len(prediction))


