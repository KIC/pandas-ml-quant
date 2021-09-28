from unittest import TestCase

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression, make_classification
from pandas_ml_common import np, pd, naive_splitter, stratified_random_splitter, FeaturesLabels
from pandas_ml_utils.ml.model.base_model import AutoEncoderModel
from pandas_ml_utils_test.ml.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils import SkModelProvider, SkAutoEncoderProvider, FittableModel, ClassificationSummary, RegressionSummary, FittingParameter
from pandas_ml_utils_test.config import DF_NOTES


class TestSkModel(TestAbstractModel):

    def test_linear_model(self):
        df = DF_NOTES.copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(Lasso()),
                    FeaturesLabels(
                        features=[
                            lambda df: df["variance"],
                            lambda df: (df["skewness"] / df["kurtosis"]).rename("engineered")
                        ],
                        labels=[
                            'authentic'
                        ]
                    )
                ),
                FittingParameter(naive_splitter()),
                verbose=1,
                callbacks=None,
                fail_silent=False
            )

        print(fit)

        prediction = df.model.predict(fit.model)
        print(prediction)

        backtest = df.model.backtest(fit.model)
        self.assertLess(backtest.model._cross_validation_models[0].sk_model.coef_[0], 1e-5)

    def test_autoencoder_model(self):
        df = DF_NOTES.copy()

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkAutoEncoderProvider([2, 1], [2],),
                    FeaturesLabels(
                        features=[
                            lambda df: df["variance"],
                            lambda df: (df["skewness"] / df["kurtosis"]).rename("engineered")
                        ],
                        labels=[
                            'authentic'
                        ]
                    )
                ),
                FittingParameter(naive_splitter()),
                verbose=1,
                callbacks=None,
                fail_silent=False
            )

        print(fit)

        prediction = df.model.predict(fit.model)
        print(prediction)

        backtest = df.model.backtest(fit.model)
        # self.assertLess(backtest.model._cross_validation_models[0].sk_model.coef_[0], 1e-5)

    #def test_partial_fit_regression(self):
    #    data = make_regression(100, 2, 1)
    #    df = pd.DataFrame(data[0])
    #    df["label"] = data[1]
    #
    #    with df.model() as m:
    #        fit_partial = m.fit(
    #            SkModel(
    #                MLPRegressor(max_iter=1, random_state=42),
    #                FeaturesAndLabels(features=[0, 1], labels=['label'])
    #            ),
    #            FittingParameter(
    #                naive_splitter(0.3),
    #                batch_size=10,
    #                fold_epochs=10
    #            )
    #        )
    #
    #    with df.model() as m:
    #        fit = m.fit(
    #            SkModel(
    #                MLPRegressor(max_iter=10, random_state=42),
    #                FeaturesAndLabels(features=[0, 1], labels=['label'])
    #            ),
    #            FittingParameter(naive_splitter(0.3))
    #        )
    #
    #    self.assertAlmostEqual(df.model.predict(fit.model).iloc[0,-1], df.model.predict(fit_partial.model).iloc[0,-1], 4)
    #
    #    self.assertEqual(len(fit_partial.model._statistics._history[('train',)]), 10)
    #    self.assertEqual(len(fit_partial.model._statistics._history[('test', 0)]), 10)
    #
    #def test_partial_fit_classification(self):
    #    data = make_classification(100, 2, 1, 0, n_clusters_per_class=1)
    #    df = pd.DataFrame(data[0])
    #    df["label"] = data[1]
    #
    #    with df.model() as m:
    #        fit_partial = m.fit(
    #            SkModel(
    #                MLPClassifier(max_iter=1, random_state=42),
    #                FeaturesAndLabels(features=[0, 1], labels=['label']),
    #                classes=np.unique(data[1])
    #            ),
    #            FittingParameter(
    #                stratified_random_splitter(0.3),
    #                batch_size=10,
    #                fold_epochs=10,
    #            )
    #        )
    #
    #    with df.model() as m:
    #        fit = m.fit(
    #            SkModel(
    #                MLPClassifier(max_iter=10, random_state=42),
    #                FeaturesAndLabels(features=[0, 1], labels=['label'])
    #            ),
    #            FittingParameter(stratified_random_splitter(0.3))
    #        )
    #
    #    self.assertAlmostEqual(df.model.predict(fit.model).iloc[0,-1], df.model.predict(fit_partial.model).iloc[0,-1], 4)

    def provide_classification_model(self, features_and_labels):
        model = SkModelProvider(
            MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001, solver='lbfgs', random_state=42),
        )

        return model

    def provide_regression_model(self, features_and_labels, **kwargs):
        model = SkModelProvider(
            MLPRegressor(1, learning_rate_init=0.01, solver='sgd', activation='identity', momentum=0, max_iter=1500, n_iter_no_change=500, nesterovs_momentum=False, shuffle=False, validation_fraction=0.0, random_state=42),
            features_and_labels,
            summary_provider=RegressionSummary,
            **kwargs
        )

        return model

    def provide_auto_encoder_model(self, features_and_labels) -> AutoEncoderModel:
        model = SkAutoEncoderProvider(
            [2, 1], [2],
            features_and_labels,
            #learning_rate_init=0.0001,
            solver='lbfgs',
            validation_fraction=0,
            activation='logistic',
            momentum=0,
            max_iter=200,
            n_iter_no_change=500,
            nesterovs_momentum=False,
            shuffle=False,
            random_state=10
        )

        return model
