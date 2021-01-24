from unittest import TestCase

from pandas_ml_quant import np
from pandas_ml_quant.technical_analysis_patch.encoder import ta_gaf, ta_rnn, ta_inverse_gasf, np_inverse_gaf
from pandas_ml_quant_test.config import DF_TEST, DF_INVERSE_GAF


class TestGAF(TestCase):

    def test_series(self):
        s = DF_TEST["Close"]
        timesteps = ta_rnn(s, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf._.values.shape
        np.testing.assert_almost_equal(0.37, gaf.iloc[-1][0][0][0], 2)
        self.assertEqual((6760, 1, 3, 3), shape)

    def test_multi_channel(self):
        df = DF_TEST[["Open", "Close"]]
        timesteps = ta_rnn(df, [1, 2, 3])

        gaf = ta_gaf(timesteps)
        shape = gaf._.values.shape
        self.assertEqual((6760, 2, 3, 3), shape)

    def test_gasf_encode_decode(self):
        s = DF_TEST["Close"]
        timesteps = ta_rnn(s, [1, 2, 3]).ta.rescale((0, 1), axis=1)

        gasf = ta_gaf(timesteps, type='invertible', rescale=True)
        shape = gasf._.values.shape

        print(shape)
        print(gasf.tail())

        print(timesteps._.values[-2:])
        print(np_inverse_gaf(gasf._.values[-2:]))

        df = ta_inverse_gasf(gasf)
        print(df.tail())
        self.assertEqual((6760, 1, 3, 3), shape)
        # np.testing.assert_array_almost_equal(s, s_rec)

    def test_inverse_gaf(self):
        df = DF_INVERSE_GAF
        inv = ta_inverse_gasf(df["prediction"])

        print(inv)
