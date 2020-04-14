from pandas_ml_common import pd, np
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.data.splitting import RandomSplits, NaiveSplitter


class TestAbstractModel(object):

    def test_classifier(self):
        """given some toy classification data"""
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0,],
            "b": [0, 0, 1, 1, 0, 0, 1, 1,],
            "c": [1, 0, 0, 1, 1, 0, 0, 1,]
        })

        """and a model"""
        model = self.provide_classification_model(FeaturesAndLabels(features=["a", "b"], labels=["c"], label_type=int))

        """when we fit the model"""
        fit = df.model.fit(model, NaiveSplitter(0.49), verbose=0, epochs=1500)
        print(fit.training_summary.df)

        prediction = df.model.predict(fit.model)
        binary_prediction = prediction.iloc[:,0] >= 0.5
        np.testing.assert_array_equal(binary_prediction, np.array([True, False, False, True, True, False, False, True,]))

        """and save and load the model"""
        # FIXME save load test -> same prediction result

    def test_regressor(self):
        """given some toy regression data"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        fit = df.model.fit(model, RandomSplits(0.3), verbose=0, epochs=500)
        print(fit.training_summary.df)

        """then we can predict"""
        prediction = df.model.predict(fit.model)
        np.testing.assert_array_almost_equal(prediction.iloc[:, 0].values, df["b"].values, 1)

        """and save and load the model"""
        # FIXME save load test -> same prediction result

    def provide_regression_model(self, features_and_labels):
        pass

    def provide_classification_model(self, features_and_labels):
        pass
