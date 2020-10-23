from unittest import TestCase

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression
from pandas_ml_common import np, pd, naive_splitter, stratified_random_splitter
from pandas_ml_utils.ml.model.base_model import AutoEncoderModel
from pandas_ml_utils_test.ml.data.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils import SkModel, SkAutoEncoderModel, FeaturesAndLabels
from pandas_ml_utils_test.config import DF_NOTES


class TestSkModel(TestAbstractModel, TestCase):

    def test_linear_model(self):
        df = DF_NOTES.copy()

        fit = df.model.fit(
            SkModel(
                Lasso(),
                FeaturesAndLabels(
                    features=[
                        lambda df: df["variance"],
                        lambda df: (df["skewness"] / df["kurtosis"]).rename("engineered")
                    ],
                    labels=[
                        'authentic'
                    ]
                )
            ),
            naive_splitter()
        )

        print(fit)

        prediction = df.model.predict(fit.model)
        print(prediction)

        backtest = df.model.backtest(fit.model)
        self.assertLess(backtest.model.sk_model.coef_[0], 1e-5)

    def test_partial_fit(self):
        data = make_regression(100, 2, 1)
        df = pd.DataFrame(data[0])
        df["label"] = data[1]

        fit_partial = df.model.fit(
            SkModel(
                MLPRegressor(max_iter=1, random_state=42),
                FeaturesAndLabels(features=[0, 1], labels=['label'])
            ),
            naive_splitter(0.3),
            batch_size=2,
            fold_epochs=10
        )

        fit = df.model.fit(
            SkModel(
                MLPRegressor(max_iter=10, random_state=42),
                FeaturesAndLabels(features=[0, 1], labels=['label'])
            ),
            naive_splitter(0.3)
        )

        self.assertAlmostEqual(df.model.predict(fit.model).iloc[0,-1], df.model.predict(fit_partial.model).iloc[0,-1], 4)
        print(len(fit.model._history))


    def provide_classification_model(self, features_and_labels):
        model = SkModel(
            MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001, solver='lbfgs', random_state=42),
            features_and_labels,
        )

        return model

    def provide_regression_model(self, features_and_labels):
        model = SkModel(
            MLPRegressor(1, learning_rate_init=0.01, solver='sgd', activation='identity', momentum=0, max_iter=1500, n_iter_no_change=500, nesterovs_momentum=False, shuffle=False, validation_fraction=0.0, random_state=42),
            features_and_labels,
        )

        return model

    def provide_auto_encoder_model(self, features_and_labels) -> AutoEncoderModel:
        model = SkAutoEncoderModel(
            [2, 1], [2],
            features_and_labels,
            learning_rate_init=0.01,
            solver='sgd',
            validation_fraction=0,
            activation='identity',
            momentum=0,
            max_iter=1500,
            n_iter_no_change=500,
            nesterovs_momentum=False,
            shuffle=False,
            random_state=42
        )

        return model