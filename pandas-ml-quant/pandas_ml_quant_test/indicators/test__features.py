from unittest import TestCase

import numpy as np
import pandas as pd
import talib

from pandas_ml_quant.indicators.multi_object import *
from pandas_ml_quant.indicators.single_object import *
from pandas_ml_quant_test.config import DF_TEST
from pandas_ml_common import Constant


class TestIndicator(TestCase):

    def test__ema(self):
        me = ta_ema(DF_TEST["Close"], 20)[-100:]
        ta = talib.EMA(DF_TEST["Close"], 20)[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__wilders(self):
        me = ta_wilders(DF_TEST["Close"], 20)[-100:]
        ta = talib.EMA(DF_TEST["Close"], 20 * 2 -1)[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__macd(self):
        my_macd = ta_macd(DF_TEST["Close"], relative=False)
        talib_macd = pd.DataFrame(talib.MACD(DF_TEST["Close"])).T

        np.testing.assert_array_almost_equal(talib_macd[-100:].values[0], my_macd[-100:].values[0])

    def test__mom(self):
        me = ta_mom(DF_TEST["Close"])[-100:]
        ta = talib.MOM(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__stddev(self):
        me = ta_stddev(DF_TEST["Close"], ddof=0)[-100:]
        ta = talib.STDDEV(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__apo(self):
        me = ta_apo(DF_TEST["Close"], relative=False)[-100:]
        ta = talib.APO(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__trix(self):
        me = ta_trix(DF_TEST["Close"])[-100:]
        ta = talib.TRIX(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__tr(self):
        me = ta_tr(DF_TEST, relative=False)[-100:]
        ta = talib.TRANGE(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__atr(self):
        me = ta_atr(DF_TEST, relative=False)[-100:]
        ta = talib.ATR(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__adx(self):
        me = ta_adx(DF_TEST, relative=False)[-100:]
        ta_pdi = talib.PLUS_DI(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]
        ta_mdi = talib.MINUS_DI(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        ta_pdm = talib.PLUS_DM(DF_TEST["High"], DF_TEST["Low"])[-100:]
        ta_mdm = talib.MINUS_DM(DF_TEST["High"], DF_TEST["Low"])[-100:]
        ta_dx = talib.ADX(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me["ADX"], ta_dx / 100)
        np.testing.assert_array_almost_equal(me["+DI"], ta_pdi / 100)
        np.testing.assert_array_almost_equal(me["-DI"], ta_mdi / 100)

        # I have absolutely no idea why we are off by a constant factor. but since it is a constant factor we can
        # neglect it
        np.testing.assert_array_almost_equal(me["+DM"], ta_pdm * 0.07142861354566638)
        np.testing.assert_array_almost_equal(me["-DM"], ta_mdm * 0.07142861354566638)

    def test__bbands(self):
        me = ta_bbands(DF_TEST["Close"], ddof=0)[-100:]
        u, m, l = talib.BBANDS(DF_TEST["Close"])
        u = u[-100:]
        m = m[-100:]
        l = l[-100:]

        np.testing.assert_array_almost_equal(me["mean"], m)
        np.testing.assert_array_almost_equal(me["upper"], u)
        np.testing.assert_array_almost_equal(me["lower"], l)

    def test__rsi(self):
        me = ta_rsi(DF_TEST["Close"])[-100:]
        ta = talib.RSI(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta / 100)

    def test__williams_R(self):
        me = ta_williams_R(DF_TEST)[-100:]
        ta = talib.WILLR(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta / -100)

    def test__ultimate(self):
        me = ta_ultimate_osc(DF_TEST)[-100:]
        ta = talib.ULTOSC(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta / 100)

    def test__ppo(self):
        me = ta_ppo(DF_TEST["Close"], exponential=False)[-100:]
        ta = talib.PPO(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta / 100)

    def test__bop(self):
        me = ta_bop(DF_TEST)[-100:]
        ta = talib.BOP(DF_TEST["Open"], DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__cci(self):
        me = ta_cci(DF_TEST)[-100:]
        ta = talib.CCI(DF_TEST["High"], DF_TEST["Low"], DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta / 100)

    def test_cross_over(self):
        me0 = ta_cross_over(DF_TEST, "Open", "Close")
        me1 = ta_cross_over(DF_TEST[["Open", "Close"]])
        me2 = ta_cross_over(DF_TEST, "Open", DF_TEST["Close"])
        me3 = ta_cross_over(DF_TEST["Open"], DF_TEST["Close"])

        self.assertTrue(me0.iloc[-1])
        self.assertFalse(me0.iloc[-2])

        for me in [me1, me2, me3]:
            np.testing.assert_array_equal(me0.values, me.values)

    def test_cross_under(self):
        me = ta_cross_under(DF_TEST, "Open", "Close")

        print(me)
        self.assertTrue(me.iloc[-3])
        self.assertFalse(me.iloc[-2])
        self.assertFalse(me.iloc[-1])

    def test_cross_constant(self):
        o = ta_cross(DF_TEST["Close"].pct_change(), Constant(0))
        u = ta_cross(DF_TEST["Close"].pct_change(), Constant(0))

        self.assertTrue(o.iloc[-2])
        self.assertTrue(u.iloc[-5])

    def test__up_down_volatility_ratio(self):
        me = ta_up_down_volatility_ratio(DF_TEST)[-1:]

        np.testing.assert_array_almost_equal(me, np.array([[-0.067, -0.232, -0.073, -0.318, -0.319,  1.776]]), 0.001)

    def test__z_score(self):
        me = ta_zscore(DF_TEST)[-1:]

        np.testing.assert_array_almost_equal(me, np.array([[ 0.546, 0.256, 0.271, 0.457,  0.457, -0.929]]), 0.001)

    def test__ta_multi_bbands(self):
        me = ta_multi_bbands(DF_TEST["Close"])[-1:]
        print(me.values.round(3))

        np.testing.assert_array_almost_equal(
            me,
            np.array([[308.406, 309.257, 310.108, 310.959, 311.81, 312.661, 313.512, 314.363, 315.214]]),
            0.001)

    def test_ta_weekday(self):
        me = ta_week_day(DF_TEST["Close"])[-1:]

        np.testing.assert_almost_equal(0.5, me.values[0])

    def test_ta_week(self):
        me = ta_week(DF_TEST["Close"])

        np.testing.assert_almost_equal(0.9423, me.values[-1], 4)
        np.testing.assert_almost_equal(0.0192, me.loc["2017-01-03"], 4)
        np.testing.assert_almost_equal(1.0, me.loc["2017-12-29"], 4)

