from typing import Tuple, Dict, Any
from unittest import TestCase

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression, make_classification
from pandas_ml_common import np, pd, naive_splitter, stratified_random_splitter, FeaturesLabels
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.model.base_model import AutoEncoderModel, ModelProvider
from pandas_ml_utils_test.ml.model.test_abstract_model import TestAbstractModel
from pandas_ml_utils import SkModelProvider, SkAutoEncoderProvider, FittableModel, ClassificationSummary, RegressionSummary, FittingParameter
from pandas_ml_utils_test.config import DF_NOTES


class TestSkModel(TestAbstractModel, TestCase):

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
                AutoEncoderModel(
                    SkAutoEncoderProvider([2, 1], [2],),
                    FeaturesLabels(
                        features=[
                            "variance",
                            "skewness"
                        ],
                        labels=[
                            'compressed'
                        ]
                    )
                ),
                FittingParameter(naive_splitter()),
                verbose=1,
                callbacks=None,
                fail_silent=False
            )

        print(fit)

        autoencoded = df.model.predict(fit.model)
        print(autoencoded.columns.tolist())

        backtest = df.model.backtest(fit.model)
        print(backtest.df.columns.tolist())

        encoded = df.model.predict(fit.model.as_encoder())
        print(encoded.columns.tolist())

        decoded = encoded[PREDICTION_COLUMN_NAME].model.predict(fit.model.as_decoder())
        print(decoded.columns.tolist())

        self.assertListEqual(
            [('prediction', 'variance'), ('prediction', 'skewness'), ('feature', 'variance'), ('feature', 'skewness')],
            autoencoded.columns.tolist()
        )
        self.assertListEqual(
            [('prediction', 'variance'), ('prediction', 'skewness'), ('label', 'variance'), ('label', 'skewness'), ('feature', 'variance'), ('feature', 'skewness')],
            backtest.df.columns.tolist()
        )
        self.assertListEqual(
            [('prediction', 'compressed'), ('feature', 'variance'), ('feature', 'skewness')],
            encoded.columns.tolist()
        )
        self.assertListEqual(
            [('prediction', 'variance'), ('prediction', 'skewness'), ('feature', 'compressed')],
            decoded.columns.tolist()
        )

    def test_partial_fit_regression(self):
        data = make_regression(100, 2, 1)
        df = pd.DataFrame(data[0])
        df["label"] = data[1]

        with df.model() as m:
            fit_partial = m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor(max_iter=1, random_state=42)),
                    FeaturesLabels(features=[0, 1], labels=['label'])
                ),
                FittingParameter(
                    naive_splitter(0.3),
                    batch_size=10,
                    fold_epochs=10
                )
            )

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(MLPRegressor(max_iter=10, random_state=42),),
                    FeaturesLabels(features=[0, 1], labels=['label'])
                ),
                FittingParameter(naive_splitter(0.3))
            )

        self.assertAlmostEqual(df.model.predict(fit.model).iloc[0,-1], df.model.predict(fit_partial.model).iloc[0,-1], 4)

        self.assertEqual(len(fit_partial.model._fit_statistics._history[('train', 'train', 'train')]), 10)
        self.assertEqual(len(fit_partial.model._fit_statistics._history[('test', 'cv 0', 'set 0')]), 10)

    def test_partial_fit_classification(self):
        data = make_classification(100, 2, 1, 0, n_clusters_per_class=1)
        df = pd.DataFrame(data[0])
        df["label"] = data[1]

        with df.model() as m:
            fit_partial = m.fit(
                FittableModel(
                    SkModelProvider(MLPClassifier(max_iter=1, random_state=42)),
                    FeaturesLabels(features=[0, 1], labels=['label']),
                    classes=np.unique(data[1])  # NOTE! Partial Fit requires to provide all classes
                ),
                FittingParameter(
                    stratified_random_splitter(0.3),
                    batch_size=10,
                    fold_epochs=10,
                )
            )

        with df.model() as m:
            fit = m.fit(
                FittableModel(
                    SkModelProvider(MLPClassifier(max_iter=10, random_state=42)),
                    FeaturesLabels(features=[0, 1], labels=['label'])
                ),
                FittingParameter(stratified_random_splitter(0.3))
            )

        self.assertAlmostEqual(df.model.predict(fit.model).iloc[0,-1], df.model.predict(fit_partial.model).iloc[0,-1], 4)

    def test_multi_sample_regressor(self):
        super().test_multi_sample_regressor()

    def test_no_test_data(self):
        super().test_no_test_data()

    def test_multindex_row(self):
        super().test_multindex_row()

    def test_multindex_row_multi_samples(self):
        super().test_multindex_row_multi_samples()

    def test_stacked_models(self):
        super().test_stacked_models()

    def test_concatenated_multi_models(self):
        super().test_concatenated_multi_models()

    def provide_classification_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        model = SkModelProvider(
            MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001, solver='lbfgs', random_state=42),
        )

        return model, dict(batch_size=None, epochs=1)

    def provide_regression_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        model = SkModelProvider(
            MLPRegressor(1, learning_rate_init=0.01, solver='sgd', activation='identity', momentum=0, max_iter=1500,
                         n_iter_no_change=500, nesterovs_momentum=False, shuffle=False, validation_fraction=0.0,
                         random_state=42)
        )

        return model, dict(batch_size=None, epochs=1)

    def provide_auto_encoder_model(self) -> Tuple[ModelProvider, Dict[str, Any]]:
        model = SkAutoEncoderProvider(
            [2, 1], [2],
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

        return model, dict(batch_size=None, epochs=1)
