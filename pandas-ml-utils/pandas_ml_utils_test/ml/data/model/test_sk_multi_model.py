from unittest import TestCase

from sklearn.neural_network import MLPClassifier

from pandas_ml_common import np, pd
from pandas_ml_utils import SkModel, FeaturesAndLabels, MultiModel
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.data.splitting import NaiveSplitter


class TestSkMultiModel(TestCase):

    def test_multi_model(self):
        """given some toy classification data"""
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0,],
            "b": [0, 0, 1, 1, 0, 0, 1, 1,],
            "c": [1, 0, 0, 1, 1, 0, 0, 1,],
            "d": [1, 0, 0, 1, 1, 0, 0, 1,],
        })

        model = MultiModel(
            SkModel(
                MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001,
                              solver='lbfgs', random_state=42),
                FeaturesAndLabels(features=["a", "b"], labels=["c", "d"], label_type=int),
            ),
            2,
            model_index_variable=None
        )

        fit = df.model.fit(model, NaiveSplitter(0.49), epochs=1500, verbose=True)
        print(fit.training_summary.df)

        self.assertEqual(4, len(fit.training_summary.df[PREDICTION_COLUMN_NAME, "c"]))
        np.testing.assert_array_almost_equal(
            fit.training_summary.df[PREDICTION_COLUMN_NAME, "c"],
            fit.training_summary.df[PREDICTION_COLUMN_NAME, "d"]
        )

    def test_multi_model_kwargs(self):
        """given some toy classification data"""
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0,],
            "b": [0, 0, 1, 1, 0, 0, 1, 1,],
            "c": [1, 0, 0, 1, 1, 0, 0, 1,],
        })

        model = MultiModel(
            SkModel(
                MLPClassifier(activation='logistic', max_iter=1000, hidden_layer_sizes=(3,), alpha=0.001,
                              solver='lbfgs', random_state=42),
                FeaturesAndLabels(
                    features=["a", "b"],
                    labels=[lambda df, index: df["c"].rename(index)],
                    label_type=int,
                    index=["z"]
                ),
            ),
            ["c1", "c2"],
            model_index_variable="index"
        )

        fit = df.model.fit(model, NaiveSplitter(0.49), epochs=1500, verbose=True)
        print(fit.training_summary.df)

        self.assertEqual(4, len(fit.training_summary.df[PREDICTION_COLUMN_NAME, "c1"]))
        self.assertEqual(4, len(fit.training_summary.df[PREDICTION_COLUMN_NAME, "c2"]))
        np.testing.assert_array_almost_equal(
            fit.training_summary.df[PREDICTION_COLUMN_NAME, "c1"],
            fit.training_summary.df[PREDICTION_COLUMN_NAME, "c2"]
        )

