import os
import tempfile
import uuid
from typing import Tuple

from pandas_ml_common import pd, np, naive_splitter, random_splitter
from pandas_ml_utils import FeaturesAndLabels, Model, SubModelFeature, FittingParameter
from pandas_ml_utils.ml.model.base_model import AutoEncoderModel
from pandas_ml_utils.ml.forecast import Forecast


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
        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

        """when we fit the model"""
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model(temp) as m:
            fit = m.fit(model, FittingParameter(splitter=naive_splitter(0.49), batch_size=batch_size, epochs=epochs), verbose=0)

        print(fit.training_summary.df)
        # fit.training_summary.df.to_pickle('/tmp/classifier.df')
        # print(fit._repr_html_())

        """then we get a html summary and can predict"""
        self.assertIn('<style>', fit.training_summary._repr_html_())

        prediction = df.model.predict(fit.model)
        binary_prediction = prediction.iloc[:,0] >= 0.5
        np.testing.assert_array_equal(binary_prediction, np.array([True, False, False, True, True, False, False, True,]))

        """and load the model"""
        try:
            copy = Model.load(temp)
            pd.testing.assert_frame_equal(df.model.predict(fit.model), df.model.predict(copy), check_less_precise=True)

            # test using context manager and ForecastProvider
            pd.testing.assert_frame_equal(
                df.model(temp).predict(forecast_provider=Forecast).df,
                df.model.predict(copy),
                check_less_precise=True
            )
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
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model, FittingParameter(splitter=naive_splitter(0.3), batch_size=batch_size, epochs=epochs),
                        verbose=0)

        print(fit.training_summary.df)
        self.assertEqual(4, len(fit.training_summary.df))
        self.assertEqual(2, len(fit.test_summary.df))

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
            "a": [1, 0] * 10,
            "b": [0, 1] * 10,
        })

        """when we fit the model"""
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model, FittingParameter(splitter=naive_splitter(0.49), batch_size=batch_size, epochs=epochs),
                        verbose=0)

        print(fit.training_summary.df)

        """then we can predict Autoencoded"""
        auto_encoded_prediction = df.model.predict(fit.model)
        self.assertEqual((20, 2), auto_encoded_prediction["prediction"].shape)

        """and we can encode"""
        encoded_prediction = df.model.predict(fit.model.as_encoder())
        print(encoded_prediction)
        self.assertEqual((20, 1), encoded_prediction["prediction"].shape)

        """and we can decode"""
        decoded_prediction = encoded_prediction["prediction"].model.predict(fit.model.as_decoder())
        print(decoded_prediction)
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
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model, FittingParameter(splitter=naive_splitter(0.3), batch_size=batch_size, epochs=epochs),
                        verbose=0)

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
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model,
                        FittingParameter(splitter=random_splitter(0.3, partition_row_multi_index=True), batch_size=batch_size, epochs=epochs),
                        verbose=0)

        prediction = df.model.predict(fit.model)
        print(fit)
        # fit.training_summary.df.to_pickle('/tmp/multi_index_row_summary.df')
        # print(fit._repr_html_())

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
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model,
                        FittingParameter(splitter=random_splitter(0.3, partition_row_multi_index=True), batch_size=batch_size, epochs=epochs),
                        verbose=0)

        self.assertEqual(8, len(fit.training_summary.df))
        self.assertEqual(4, len(fit.test_summary.df))

        prediction = df.model.predict(fit.model, samples=2)
        self.assertEqual(2, len(prediction.iloc[:, 0]._.values))
        self.assertEqual((6, 2), prediction.loc["A"].iloc[:, 0]._.values.shape)
        self.assertEqual((6, 2), prediction.loc["B"].iloc[:, 0]._.values.shape)

    def test_no_test_data(self):
        """given some toy regression data"""
        df = pd.DataFrame({
            "a": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]
        })

        """and a model"""
        model = self.provide_regression_model(FeaturesAndLabels(features=["a"], labels=["b"]))

        """when we fit the model"""
        batch_size, epochs = self.provide_batch_size_and_epoch()
        with df.model() as m:
            fit = m.fit(model,
                        FittingParameter(splitter=naive_splitter(0), batch_size=batch_size, epochs=epochs),
                        verbose=0)

        # print(fit.training_summary.df)
        print(fit.test_summary.df)

        """then we have an empty test data frame"""
        self.assertEqual(len(fit.training_summary.df), len(df))
        self.assertEqual(len(fit.test_summary.df), 0)

    def test_stacked_models(self):
        """given some toy classification data"""
        df = pd.DataFrame({
            "a": [1, 0, 1, 0, 1, 0, 1, 0, ],
            "b": [0, 0, 1, 1, 0, 0, 1, 1, ],
            "c": [1, 0, 0, 1, 1, 0, 0, 1, ]
        })

        """and a model"""
        model = self.provide_classification_model(
            FeaturesAndLabels(
                features=[
                    "a",
                    SubModelFeature("b", self.provide_classification_model(
                        FeaturesAndLabels(features=["a", "b"], labels=["c"], label_type=int)
                    ))
                ],
                labels=["c"],
                label_type=int
            )
        )

        temp = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

        with self.assertLogs(level='INFO') as cm:
            with df.model(temp) as m:
                fit = m.fit(model)

            self.assertIn("INFO:pandas_ml_utils.ml.model.base_model:fitting submodel: b", cm.output[0])
            self.assertIn(
                "INFO:pandas_ml_utils.ml.model.base_model:fitted submodel",
                [s for s in cm.output if s.startswith("INFO:pandas_ml_utils.ml.model.base_model:fitted")][0]
            )

        prediction = df.model.predict(fit.model)
        prediction2 = df.model.predict(Model.load(temp))
        pd.testing.assert_frame_equal(prediction, prediction2)
        os.remove(temp)

    # Abstract methods
    def provide_batch_size_and_epoch(self) -> Tuple[int, int]:
        return (None, 1)

    def provide_regression_model(self, features_and_labels) -> Model:
        pass

    def provide_classification_model(self, features_and_labels) -> Model:
        pass

    def provide_auto_encoder_model(self, features_and_labels) -> AutoEncoderModel:
        return None