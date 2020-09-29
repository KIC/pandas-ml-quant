import os
import tempfile
import uuid

from pandas_ml_common import pd, np, naive_splitter, random_splitter
from pandas_ml_utils import FeaturesAndLabels, Model
from pandas_ml_utils.ml.model.auto_encoder_model import AutoEncoderModel


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
        fit = df.model.fit(model, naive_splitter(0.49), verbose=0, epochs=1500)
        print(fit.training_summary.df)

        prediction = df.model.predict(fit.model)
        binary_prediction = prediction.iloc[:,0] >= 0.5
        np.testing.assert_array_equal(binary_prediction, np.array([True, False, False, True, True, False, False, True,]))

        """and save and load the model"""
        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        try:
            fit.model.save(temp)
            copy = Model.load(temp)
            pd.testing.assert_frame_equal(df.model.predict(fit.model), df.model.predict(copy), check_less_precise=True)
        finally:
            os.remove(temp)

    def test_regressor(self):
        """given some toy regression data"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        fit = df.model.fit(model, random_splitter(0.3), verbose=0, epochs=500)
        print(fit.training_summary.df)

        """then we can predict"""
        prediction = df.model.predict(fit.model)
        np.testing.assert_array_almost_equal(prediction.iloc[:, 0].values, df["b"].values, 1)

        """and save and load the model"""
        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        try:
            fit.model.save(temp)
            copy = Model.load(temp)
            pd.testing.assert_frame_equal(df.model.predict(fit.model), df.model.predict(copy), check_less_precise=True)
        finally:
            os.remove(temp)

    def test_auto_encoder(self):
        """given the implementation can handle auto encoders"""
        model = self.provide_auto_encoder_model(
            FeaturesAndLabels(
                features=["a", "b"],
                labels=["a", "b"],
                latent=["x"]
            )
        )

        if model is None:
            return

        """and some toy classification data"""
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, ],
            "b": [0, 1, 0, 1, ],
        })

        """when we fit the model"""
        fit = df.model.fit(model, naive_splitter(0.49), verbose=0, epochs=500)
        print(fit.training_summary.df)

        """then we can encoder"""
        encoded_prediction = df.model.predict(fit.model.as_encoder())
        print(encoded_prediction)
        self.assertEqual((4, 1), encoded_prediction["prediction"].shape)

        """and we can decoder"""
        decoded_prediction = encoded_prediction.model.predict(fit.model.as_decoder())
        print(decoded_prediction)
        np.testing.assert_array_almost_equal(decoded_prediction["prediction"].values[:2], fit.training_summary.df["prediction"].values, 1)
        np.testing.assert_array_almost_equal(decoded_prediction["prediction"].values > 0.5, df[["a", "b"]].values)

        """and we can encoder and decode after safe and load"""
        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        try:
            fit.model.save(temp)
            copy = Model.load(temp)

            pd.testing.assert_frame_equal(
                df.model.predict(fit.model.as_encoder()),
                df.model.predict(copy.as_encoder()),
                check_less_precise=True)

            pd.testing.assert_frame_equal(
                encoded_prediction.model.predict(fit.model.as_decoder()),
                encoded_prediction.model.predict(copy.as_decoder()),
                check_less_precise=True)
        finally:
            os.remove(temp)

        # try to save only as encoder model
        try:
            fit.model.as_encoder().save(temp)
            copy = Model.load(temp)
        finally:
            os.remove(temp)

    def test_multi_sample_regressor(self):
        """given some toy regression data"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        fit = df.model.fit(model, random_splitter(0.3), verbose=0, epochs=500)
        print(fit.training_summary.df)

        """then we can predict"""
        prediction = df.model.predict(fit.model, samples=2)
        np.testing.assert_array_almost_equal(prediction.iloc[:, 0]._.values, np.concatenate([df[["b"]].values, df[["b"]].values], axis=1), 1)

    def test_multindex_row(self):
        """given some toy regression data while we provide a multiindex for the rows"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0, -2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        }, index=pd.MultiIndex.from_product([["A", "B"], range(6)]))

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        fit = df.model.fit(model, random_splitter(0.3), verbose=0, epochs=500)
        prediction = df.model.predict(fit.model)

        """then we get a prediction for A and B rows"""
        self.assertEqual(8, len(fit.training_summary.df))
        self.assertEqual(4, len(fit.training_summary.df.loc["A"]))
        self.assertEqual(4, len(fit.training_summary.df.loc["B"]))

        self.assertEqual(4, len(fit.test_summary.df))
        self.assertEqual(2, len(fit.test_summary.df.loc["A"]))
        self.assertEqual(2, len(fit.test_summary.df.loc["B"]))

        self.assertEqual(6, len(prediction.loc["A"]))
        self.assertEqual(6, len(prediction.loc["B"]))
        np.testing.assert_array_almost_equal(prediction.iloc[:, 0].values, df["b"].values, 1)

    def test_multindex_row_multi_samples(self):
        """given some toy regression data while we provide a multiindex for the rows"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0, -2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        }, index=pd.MultiIndex.from_product([["A", "B"], range(6)]))

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        fit = df.model.fit(model, random_splitter(0.3), verbose=0, epochs=500)

        self.assertEqual(8, len(fit.training_summary.df))
        self.assertEqual(4, len(fit.test_summary.df))

        prediction = df.model.predict(fit.model, samples=2)
        self.assertEqual((12, 2), prediction.iloc[:, 0]._.values.shape)
        self.assertEqual(2, fit.model._history.shape[1])
        self.assertGreater(fit.model._history.shape[0], 2)

    # Abstract methods
    def provide_regression_model(self, features_and_labels) -> Model:
        pass

    def provide_classification_model(self, features_and_labels) -> Model:
        pass

    def provide_auto_encoder_model(self, features_and_labels) -> AutoEncoderModel:
        return None