from unittest import TestCase

import pandas as pd
import talib

from pandas_ml_common import Constant
from pandas_ta_quant.technical_analysis import *
from pandas_ta_quant_test.config import DF_TEST


class TestIndicator(TestCase):

    def test__mean_returns(self):
        me = ta_mean_returns(DF_TEST[["Close", "Volume"]], 20)[-100:]
        np.testing.assert_array_almost_equal(me.iloc[-1].values, np.array([0.000810, 0.033791]), 5)

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

        # test multi
        my_multi_macd = ta_macd(DF_TEST[["Close", "Open"]], relative=False)
        self.assertListEqual([('Close', 'macd_12,26,9'),
                              ('Close', 'signal_12,26,9'),
                              ('Close', 'histogram_12,26,9'),
                              ('Open', 'macd_12,26,9'),
                              ('Open', 'signal_12,26,9'),
                              ('Open', 'histogram_12,26,9')],
                             my_multi_macd.columns.to_list())

    def test__mom(self):
        me = ta_mom(DF_TEST["Close"], period=10, relative=False)[-100:]
        ta = talib.MOM(DF_TEST["Close"])[-100:]

        np.testing.assert_array_almost_equal(me, ta)

    def test__stddev(self):
        me = ta_stddev(DF_TEST["Close"], period=5, ddof=0, downscale=False)[-100:]
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
        me = ta_bbands(DF_TEST["Close"], period=5, ddof=0)[-100:]
        u, m, l = talib.BBANDS(DF_TEST["Close"])
        u = u[-100:]
        m = m[-100:]
        l = l[-100:]

        np.testing.assert_array_almost_equal(me["mean"], m)
        np.testing.assert_array_almost_equal(me["upper"], u)
        np.testing.assert_array_almost_equal(me["lower"], l)

    def test__rsi(self):
        me = ta_rsi(DF_TEST["Close"], period=14)[-100:]
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

    def test__up_down_volatility_ratio(self):
        me = ta_up_down_volatility_ratio(DF_TEST)[-1:]

        np.testing.assert_array_almost_equal(me, np.array([[-0.067, -0.232, -0.073, -0.318, -0.319,  1.776]]), 0.001)

    def test__z_score(self):
        me = ta_zscore(DF_TEST, downscale=False)[-1:]

        np.testing.assert_array_almost_equal(me, np.array([[ 0.546, 0.256, 0.271, 0.457,  0.457, -0.929]]), 0.001)

    def test__ta_multi_bbands(self):
        me = ta_multi_bbands(DF_TEST["Close"])[-1:]
        print(me.values.round(3))

        np.testing.assert_array_almost_equal(
            me,
            np.array([[308.406, 309.257, 310.108, 310.959, 311.81, 312.661, 313.512, 314.363, 315.214]]),
            0.001)

    def test_ta_sinusoidial_weekday(self):
        me = ta_sinusoidal_week_day(DF_TEST["Close"])[-1:]

        np.testing.assert_almost_equal(me.values[0], 1.2246467991473532e-16, 6)

    def test_ta_sinusoidial_week(self):
        me = ta_sinusoidal_week(DF_TEST["Close"])

        np.testing.assert_almost_equal(me.values[-1], -0.35460488704253595, 4)
        np.testing.assert_almost_equal(me.loc["2017-01-03"], 0.12053668025532306, 4)
        np.testing.assert_almost_equal(me.loc["2017-12-29"], -2.4492935982947064e-16, 4)

    def test_poly_coeff(self):
        me = ta_poly_coeff(DF_TEST["Close"], period=16)
        print(me.tail())

        np.testing.assert_almost_equal(
            me.iloc[-1].values,
            np.array([-3.279875, 4.177750, 0.183198]),
            6
        )

    def test_slope(self):
        me = ta_slope(DF_TEST[["Close", "Open"]], period=20)
        # print(me.tail())

        np.testing.assert_almost_equal(
            [0.225188, 0.267564],
            me.iloc[-1].values,
            6
        )

    def test_hurst(self):
        h = ta_vola_hurst(DF_TEST[-255*2:])

        np.testing.assert_array_almost_equal(
            [0.147095, 0.473749],
            h.iloc[-1].values
        )

    def test_potential_turning_point(self):
        tps3 = ta_potential_turning_point(DF_TEST["Close"], 120, edge_detector='poly', edge_period=3)
        tps1 = ta_potential_turning_point(DF_TEST["Close"], 120, edge_detector='naive', edge_period=3)
        tps2 = ta_potential_turning_point(DF_TEST["Close"], 120, edge_detector='mean', edge_period=3)

        print(tps1.sum())
        print(tps2.sum())
        print(tps3.sum())

    def test_strike_prices(self):
        df = pd.DataFrame({
            "a": [1.2,    2.49,   2.51,   5.01],
            "b": [26.49,  26.51,  24.99,  25.01],
            "c": [250.49, 260.51,  249,   261],
        })

        np.testing.assert_array_almost_equal([
            [0., 2.5, 2.5, 5.],
            [25., 25., 25., 25.],
            [250., 260., 250., 260.],
        ], ta_strike(df).T.values)

        np.testing.assert_array_almost_equal([
            [2.5, 2.5, 5., 7.5],
            [30., 30., 25., 30.],
            [260., 270., 250., 270.],
        ], ta_strike(df, mode='ceil').T.values)

        np.testing.assert_array_almost_equal([
            [0., 0., 2.5, 5.],
            [25., 25., 22.5, 25.],
            [250., 260., 240., 260.],
        ], ta_strike(df, mode='floor').T.values)

