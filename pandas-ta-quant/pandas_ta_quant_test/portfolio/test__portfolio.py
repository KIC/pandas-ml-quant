from csv import reader
from unittest import TestCase

import numpy
import numpy as np
import pandas as pd

from pandas_ta_quant.portfolio import Portfolio, TargetQuantity, TargetWeight
from pandas_ta_quant_test.config import CSV_TRADES


class TestPortfolio(TestCase):

    def get_trades(self):
        with open(CSV_TRADES) as csv:
            csv_reader = reader(csv)
            header = next(csv_reader)

            for trade in csv_reader:
                if trade[0] == 'VXX':
                    strategy = 'intraday'
                elif trade[0] == 'SPY':
                    strategy = 'slow'
                elif trade[0] == 'WPG':
                    strategy = 'options'
                else:
                    strategy = None

                yield trade[0], trade[1], trade[2], trade[4], trade[5], float(trade[7]), float(trade[8]), None, trade[6], strategy, trade[3], float(trade[-1])

    def test_orders_with_nav(self):
        p = Portfolio()

        for trade in self.get_trades():
            p.trade(*trade)

        pf = p.get_current_portfolio()

        numpy.testing.assert_array_almost_equal(
            pf.loc["WPG"].sum().values,
            np.array([198., 465., -2.973, -np.inf]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["SPY"].sum().values,
            np.array([0., -20., -0.025, -np.inf]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["VXX"].sum().values,
            np.array([0., 39., -0.001, -np.inf]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["USD"].sum().values,
            np.array([-484.0 -2.999, -484.0 -2.999, 0, -484.0 -2.999]),
            decimal=3
        )

    def test_orders_with_price_and_quantity(self):
        p = Portfolio()

        p.trade('MSFT', 'MSFT', '2021-01-01 00:00:00', 'B', None, 10, price=2.5, fee=-1)
        p.trade('MSFT', 'MSFT', '2021-01-01 00:00:00', 'B', None, -5, price=2, fee=-1)
        pf = p.get_current_portfolio()
        # print(pf)

        numpy.testing.assert_array_almost_equal(
            pf.loc['USD'].sum().values,
            np.array([-25 +10 -2, -25 +10 -2, 0, -25 +10 -2]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["MSFT"].sum().values,
            np.array([5, 15, -2, -np.inf]),
            decimal=3
        )

    def test_orders_with_price_and_target_quantity(self):
        p = Portfolio()

        p.trade('MSFT', 'MSFT', '2021-01-01 00:00:00', 'B', None, TargetQuantity(10), price=2.5, fee=-1)
        p.trade('MSFT', 'MSFT', '2021-01-01 00:00:00', 'B', None, TargetQuantity(5), price=2, fee=-1)
        pf = p.get_current_portfolio()
        #print(pf)

        numpy.testing.assert_array_almost_equal(
            pf.loc['USD'].sum().values,
            np.array([-25 + 10 - 2, -25 + 10 - 2, 0, -25 + 10 - 2]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["MSFT"].sum().values,
            np.array([5, 15, -2, -np.inf]),
            decimal=3
        )

    def test_orders_with_price_and_target_weight(self):
        p = Portfolio(capital=100)
        p.price.push_quote('MSFT', '2021-01-01 00:00:00', bid=10, ask=10)
        p.price.push_quote('AAPL', '2021-01-01 00:00:00', bid=0.1, ask=0.1)
        p.price.push_quote('VXX', '2021-01-01 00:00:00', bid=0.1, ask=0.1)
        p.price.push_quote('VXY', '2021-01-01 00:00:00', bid=0.1, ask=0.1)

        p.trade('MSFT', 'MSFT', '2021-01-02 00:00:00', 'B', None, TargetWeight(0.3), fee=0)
        p.trade('AAPL', 'AAPL', '2021-01-02 00:00:00', 'B', None, TargetWeight(0.7), fee=0)
        p.trade('VXX', 'VXX', '2021-01-02 00:00:00', 'B', None, TargetWeight(-0.2), fee=0)
        p.trade('VXY', 'VXY', '2021-01-02 00:00:00', 'B', None, TargetWeight(0.2), fee=0)

        pf = p.get_current_portfolio()
        #print(pf)

        numpy.testing.assert_array_almost_equal(
            pf["nav"].values,
            np.array([70, 30, 0, -20, 20]),
            decimal=3
        )

    def test_orders_with_price_and_target_weight_rebalance(self):
        # test bid/ask
        p = Portfolio(capital=100)
        p.price.push_quote('MSFT', '2021-01-01 00:00:00', bid=9.99, ask=10)
        p.trade('MSFT', 'MSFT', '2021-01-02 00:00:00', 'B', None, TargetWeight(0.3), fee=0)
        p.trade('MSFT', 'MSFT', '2021-01-02 00:00:00', 'B', None, TargetWeight(0.6), fee=0)
        pf = p.get_current_portfolio()
        print(pf)

        numpy.testing.assert_array_almost_equal(
            pf[["nav", "liquidation_value"]].values.T,
            np.array([[60.012, 39.988],
                      [59.951988, 39.988000]]),
            decimal=3
        )

    def test_with_quantity_no_price(self):
        p = Portfolio()

        #  price = 2.5
        p.price.push_quote('MSFT', '2021-01-02 00:00:00', bid=2, ask=2.5)
        p.trade('MSFT', 'MSFT', '2021-01-02 00:00:01', 'B', None, 10, fee=-1)

        p.price.push_quote('MSFT', '2021-01-03 00:00:00', bid=2, ask=2.5)
        p.trade('MSFT', 'MSFT', '2021-01-03 00:00:01', 'B', None, TargetQuantity(5), fee=-1)
        pf = p.get_current_portfolio()
        # print(pf)

        numpy.testing.assert_array_almost_equal(
            pf.loc['USD'].sum().values,
            np.array([-25 + 10 - 2, -25 + 10 - 2, 0, -25 + 10 - 2]),
            decimal=3
        )
        numpy.testing.assert_array_almost_equal(
            pf.loc["MSFT"].sum().values,
            np.array([5, 15, -2, 10]),
            decimal=3
        )

    def test_lala(self):
        p = Portfolio()

        for trade in self.get_trades():
            p.trade(*trade)

        pf = p.get_current_portfolio()
        print(pf)

        pf2 = p.foo(False)
        print(pf2)
        numpy.testing.assert_array_almost_equal(pf.values, pf2.values)

        pf3 = p.foo(True)
        print(pf3)

        numpy.testing.assert_array_almost_equal(p.get_portfolio_timeseries().values[:,:3], pf3.values[:,:3])