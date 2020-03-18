from unittest import TestCase
from pandas_ml_quant import pd, np
from pandas_ml_quant.encoders.one_hot import ta_one_hot_encode_discrete
from pandas_ml_quant_test.config import DF_TEST


class TestOneHotEncoder(TestCase):

    def test_one_hot_encoder_vec(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 4, 5]})

        encoded = ta_one_hot_encode_discrete(df["a"])

        print(repr(encoded))
        self.assertListEqual(df.index.tolist(), encoded.index.tolist())
        np.testing.assert_array_equal(np.array([[1, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0],
                                                [0, 0, 1, 0, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 1]]),
                                      encoded.ml.values)

    def test_one_hot_encoder_arr(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 4, 5],
                           "b": [1, 2, 3, 4, 4, 5]})

        encoded = ta_one_hot_encode_discrete(df[["a", "b"]])

        print(repr(encoded.ml.values))
        self.assertListEqual(df.index.tolist(), encoded.index.tolist())
        np.testing.assert_array_equal(np.array([[[1, 0, 0, 0, 0],
                                                 [1, 0, 0, 0, 0]],

                                                [[0, 1, 0, 0, 0],
                                                 [0, 1, 0, 0, 0]],

                                                [[0, 0, 1, 0, 0],
                                                 [0, 0, 1, 0, 0]],

                                                [[0, 0, 0, 1, 0],
                                                 [0, 0, 0, 1, 0]],

                                                [[0, 0, 0, 1, 0],
                                                 [0, 0, 0, 1, 0]],

                                                [[0, 0, 0, 0, 1],
                                                 [0, 0, 0, 0, 1]]]),
                                      encoded.ml.values)

    def test_chained_encoder(self):
        df = DF_TEST.copy()

        discrete = df["Close"].q.ta_future_bband_quantile(7, period=14).dropna()
        onehot = df["Close"].q.ta_future_bband_quantile(7, period=14).q.ta_one_hot_encode_discrete()

        self.assertEqual(len(discrete), len(onehot))
        self.assertEqual(1., discrete[-1:].values[0])
        self.assertListEqual([0, 1, 0], onehot.iloc[-1])