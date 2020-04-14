from pandas_ml_common import pd, np
from pandas_ml_utils.ml.data.splitting import RandomSplits


class TestAbstractModel(object):

    def test_classifier(self):
        # test safe and load
        pass

    def test_regressor(self):
        """given some toy data"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        """and a model"""
        model = self.provide_regression_model()

        """when we fit the model"""
        fit = df.model.fit(model, RandomSplits(0.3), verbose=0, epochs=500)
        print(fit.training_summary.df)

        """then we can predict"""
        prediction = df.model.predict(fit.model)
        np.testing.assert_array_almost_equal(prediction.iloc[:, 0].values, df["b"].values, 1)

        """and save and load the model"""


    def provide_regression_model(self):
        pass