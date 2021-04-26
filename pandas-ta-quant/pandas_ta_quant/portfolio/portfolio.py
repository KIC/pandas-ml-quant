import numbers
import uuid
from collections import namedtuple, defaultdict
from io import StringIO
from typing import Tuple, Union, Callable, List
import logging

import numpy as np
import pandas as pd

from pandas_ml_common.utils.time_utils import parse_timestamp, min_timestamp
from .price import PriceTimeSeries

TrxKey = namedtuple('TrxKey', ['underlying', 'instrument', 'timestamp'])
Order = namedtuple('Order', ['ID', 'order_type', 'order_subtype', 'currency', 'quantity', 'nav', 'fee', 'strategy'])
_log = logging.getLogger(__name__)


class Quantity(object):

    def __init__(self, v: float):
        self.v = v

    def value(self,
              lazy_current_portfolio: Callable[[str, str], pd.DataFrame],
              underlying: str,
              instrument: str,
              currency: str,
              prices: PriceTimeSeries,
              tst: pd.Timestamp,
              price: float = None,
              fee: float = 0,
             ):
        return self.v


class TargetQuantity(Quantity):

    def __init__(self, v: float):
        super().__init__(v)

    def value(self,
              lazy_current_portfolio: Callable[[str, str], pd.DataFrame],
              underlying: str,
              instrument: str,
              currency: str,
              prices: PriceTimeSeries,
              tst: pd.Timestamp,
              price: float = None,
              fee: float = 0,
             ):
        current_position = lazy_current_portfolio(underlying, instrument)
        if current_position.empty:
            return self.v
        else:
            current_quantity = current_position['quantity'].item()
            quantity = self.v - current_quantity
            return quantity


class TargetWeight(Quantity):

    def __init__(self, v: float):
        assert -1 <= v <= 1, 'Weighs need to be between -1 and 1'
        super().__init__(v)
        self.warned_fee = False

    def value(self,
              lazy_current_portfolio: Callable[[str, str], pd.DataFrame],
              underlying: str,
              instrument: str,
              currency: str,
              prices: PriceTimeSeries,
              tst: pd.Timestamp,
              price: float = None,
              fee: float = 0,
             ):
        if fee != 0:
            if not self.warned_fee:
                _log.warning("we can't handle fees properly using TargetWeight. Your PriceTimeseries should use bid/ask or spread!")
                self.warned_fee = True

        current_portfolio = lazy_current_portfolio(None, None)
        try:
            current_cash_position = current_portfolio.loc[currency]["nav"].sum()
        except KeyError:
            raise ValueError('Portfolio needs to be funded `Portfolio(capital=100)`')

        current_positions = current_portfolio.drop([currency])[["liquidation_value"]].sum().item()
        target_nav = (current_cash_position + current_positions) * self.v

        idx = (underlying, instrument)

        current_nav = current_portfolio.loc[idx]["liquidation_value"].sum().item() if idx in current_portfolio.index else 0
        if current_nav <= -np.inf:
            raise ValueError(f"TargetWeight needs prices! Missing price for {idx}")

        if price is not None:
            qty = (target_nav - current_nav) / price
        else:
            _, (bid, ask) = prices.get_price(instrument, tst, currency)
            price = bid if target_nav < current_nav else ask
            qty = (target_nav - current_nav) / price

        return qty


class Portfolio(object):

    @staticmethod
    def from_tradelog_file(file: Union[str, List[str]], *args, strategy_mapper: Callable[[TrxKey], str] = None, **kwargs):
        trandelog_files = [file] if not isinstance(file, (list, tuple)) else file
        content = defaultdict(list)

        for tradelog_file in trandelog_files:
            for line in open(tradelog_file).readlines():
                if len(line) <= 1:
                    continue

                if '|' not in line:
                    key = line.replace('\n', '')
                    continue

                content[key].append(line)

        log = {k: pd.read_csv(StringIO('\n'.join(v)), sep='|', parse_dates=True, header=None) for k, v in content.items()}
        transactions = {k: v for k, v in log.items() if k.endswith('_TRANSACTIONS')}

        pf = Portfolio(*args, **kwargs)
        for asset, trx_asset in transactions.items():
            for i, trx in trx_asset.iterrows():
                # transaction key
                if asset.startswith('STOCK'):
                    trx_key = TrxKey(trx[2], trx[2], pd.to_datetime(f'{trx[7]} {trx[8]}', format='%Y%m%d %H:%M:%S'))
                elif asset.startswith('OPTION'):
                    trx_key = TrxKey(trx[2].split(' ')[0], trx[3], pd.to_datetime(f'{trx[7]} {trx[8]}', format='%Y%m%d %H:%M:%S'))
                elif asset.startswith('FUTURE'):
                    trx_key = TrxKey(trx[3].split(' ')[0], trx[3], pd.to_datetime(f'{trx[7]} {trx[8]}', format='%Y%m%d %H:%M:%S'))
                else:
                    _log.warning(f'Not implemented {asset}')
                    continue

                # order type and sub type
                if trx[6] in ['A;C']:
                    ot = list(reversed(trx[6].split(';')))
                else:
                    ot = trx[6].split(';') + [None]

                strategy = None if strategy_mapper is None else strategy_mapper(trx_key)
                order = Order(trx[1], ot[0], ot[1], trx[9], trx[10], trx[13], trx[14], strategy)

                pf.trade(trx_key.underlying, trx_key.instrument, trx_key.timestamp,
                         order.order_type, order.order_subtype, order.quantity, order.nav, fee=order.fee,
                         strategy=order.strategy)

        return pf

    def __init__(self, prices: PriceTimeSeries = None, capital: float = None, currency: str = 'USD'):
        self.prices = PriceTimeSeries() if prices is None else prices
        self.currency = currency
        self.orders = []

        if capital is not None:
            self.trade(currency, currency, min_timestamp(), 'F', None, nav=capital, price=1, currency=currency, id=str(uuid.uuid4()))

    def trade(self,
              underlying: str,
              instrument: str,
              timestamp: Union[pd.Timestamp, str, Tuple[str, str]],
              order_type: str,
              order_subtype: str = None,
              quantity: Union[float, Quantity] = None,
              nav: float = None,
              price: float = None,
              currency: str = None,
              strategy: str = None,
              id: str = str(uuid.uuid4()),
              fee: float = 0,
             ):
        """
        :param underlying: any string
        :param instrument: any string, can be None we then just use underlying
        :param timestamp: any timestamp - NOTE it is important to pass a time (in UTC)!
            if you want to buy at "close" use 23:59:59.9999
        :param order_type: BO (Buy to Open), SC (Sell to Close), SO (Sell to Open), BC (Buy to Close), A (Assigned)
        :param order_subtype: None, E (Expired), O (Open), A (Assigned)
        :param quantity: any float
        :param nav: None or any float
        :param price: None,
            if price is given and nav is None, we calculate the nav using quantity * price
            if price is None and nav is None, we need a price lookup table and a lookup rule i.e. bid/ask, open/close, ...
                if no price lookup table provided throw exception
                but the whole API can be lazy such that we need prices only on get_trades_dataframe()
        :param fee: 0, or any flot or callable(nav, quantity)
            if fee is callable we calculate the fee using the qunatinty and nav value
        :param currency: USD, or any string
        :param strategy: None, or any sting (used as an analytics field)
        :param id: UUID or any other string/primitive

        :return:
        """
        assert nav is not None or quantity is not None, "Quantity or Net Asset Value (nav) need to be passed!"
        assert currency is None or currency == self.currency, "Multiple currencies are not supported at the moment"

        # parse timestamp
        tst = parse_timestamp(timestamp)

        # assign default currency
        if currency is None:
            currency = self.currency

        # calculate net asset value
        if nav is None:
            # calculate quantity
            if isinstance(quantity, numbers.Number):
                quantity = Quantity(quantity)

            quantity = quantity.value(lambda u, i: self.get_current_position(u, i),
                                      underlying, instrument, self.currency, self.prices, tst, price, fee)

            if price is None:
                price_tst, (bid, ask) = self.prices.get_price(instrument, tst, currency)
                price = bid if quantity < 0 else ask

            # we calculate the nav using quantity * price
            nav = quantity * price
        elif quantity is None:
            # calculate quantity
            if price is None:
                price_tst, (bid, ask) = self.prices.get_price(instrument, tst, currency)
                price = bid if nav < 0 else ask

            quantity = nav / price

        order = Order(id, order_type, order_subtype, currency, quantity, nav, fee, strategy)
        trx_key = TrxKey(underlying, underlying if instrument is None else instrument, tst)
        self.orders.append(pd.Series(order, index=order._fields, name=trx_key))

        if instrument != currency:
            # now also make a cash trade
            #  TODO later when we support multiple currencies also make an FX-Trade the very same way
            #   Note that we can not aggregate over different currencies anymore
            cash_ot = 'P' if nav > 0 else 'R'
            cash_id = f'{id}_{currency}'
            self.trade(currency, currency, timestamp, cash_ot, 'T', nav=-nav+fee, price=1, currency=currency, strategy=strategy, id=cash_id)

    @property
    def price(self):
        return self.prices

    def get_current_portfolio(self, strategy=None):
        return self.get_current_position(None, None, strategy)

    def get_current_position(self, underlying, instrument=None, strategy=None):
        index = (underlying, ) if instrument is None else (underlying, instrument)
        df_trades = self.get_order_sequence().drop(['order_type', 'order_subtype', 'strategy'], axis=1)

        if strategy is not None:
            df_trades = df_trades[df_trades["strategy"] == strategy]

        def evaluate_liquidation_value(pos: pd.Series):
            # TODO move this function outside such that anyone can call it
            #   allow passing an aggregat function such that we can sum or cumsum
            #   further group the positions inside this function among currencies
            #   create constants for nav and liquidation value such taht we can easily rename them later
            res = pos[['quantity', 'nav', 'fee']].sum()
            tst = pos.index.to_list()[-1]
            if isinstance(tst, tuple):
                tst = tst[-1]

            inst = pos.name[1] if instrument is None else instrument

            try:
                val = res['nav'] if inst == self.currency else \
                        self.prices.get_price(inst, tst, pos["currency"].values[-1])[1][0] * res["quantity"].item()
                res["liquidation_value"] = val
            except KeyError:
                res["liquidation_value"] = -np.inf
                # _log.error(f"Failed to get price for {inst} @ {tst}")

            if pos["currency"].values[-1] != self.currency:
                # we need to get an FX rate as well
                raise NotImplementedError("Multiple currencies not supported at the moment")

            return res

        if underlying is None and instrument is None:
            return df_trades.groupby(level=[0, 1]).apply(evaluate_liquidation_value)
        else:
            try:
                return df_trades.loc[index]\
                    .groupby(level=range(df_trades.index.nlevels - len(index)))\
                    .apply(evaluate_liquidation_value)
            except KeyError:
                return pd.DataFrame({}, df_trades.columns)

    def get_portfolio_timeseries(self):
        df_trades = self.get_order_sequence()
        # TODO we need a similr function like the `evaluate_liquidation_value` but with a cumsum like approach
        return df_trades[['quantity', 'nav', 'fee']].groupby(level=[0, 1]).transform(pd.DataFrame.cumsum)

    def get_order_sequence(self) -> pd.DataFrame:
        df_trades = pd.DataFrame(self.orders, columns=Order._fields)
        df_trades.index = pd.MultiIndex.from_tuples(df_trades.index, names=TrxKey._fields)
        df_trades = df_trades.drop(['ID'], axis=1).sort_index()
        return df_trades

    def foo(self, ts):
        df_trades = self.get_order_sequence()
        if ts:
            return df_trades.groupby(level=[0, 1]).apply(get_position_timeseries_aggregator(self.prices, None, self.currency))
        else:
            return df_trades.groupby(level=[0, 1]).apply(get_position_aggregator(self.prices, None, self.currency))


def get_position_aggregator(prices: PriceTimeSeries, instrument: str, base_currency: str):
    def aggregator(pos: pd.Series):
        res = pos[['quantity', 'nav', 'fee', 'currency']].groupby('currency')
        res = res.sum()

        tst = pos.index.get_level_values(-1).max()
        inst = pos.name[1] if instrument is None else instrument

        def mark_to_market(pos):
            try:
                val = res['nav'] if inst == base_currency else \
                    prices.get_price(inst, tst, pos["currency"].item())[1][0] * pos["quantity"].item()
            except KeyError:
                val = -np.inf
                # _log.error(f"Failed to get price for {inst} @ {tst}")

            # look if we need a different currency
            if pos.name != base_currency:
                # we need to get an FX rate as well
                raise NotImplementedError("Multiple currencies not supported at the moment")

            return val

        res["liquidation_value"] = res.apply(mark_to_market, axis=1)
        return res

    return aggregator


def get_position_timeseries_aggregator(prices: PriceTimeSeries, instrument: str, base_currency: str):
    def aggregator(pos: pd.Series):
        grp = pos[['quantity', 'nav', 'fee', 'currency']].groupby('currency')
        grp = {g: f.drop('currency', axis=1).cumsum() for g, f in grp}
        res = pd.concat(grp.values(), axis=0, keys=grp.keys())

        def mark_to_market(pos):
            try:
                val = pos['nav'] if pos.name[1] == base_currency else \
                    prices.get_price(pos.name[1], pos.name[-1], pos["currency"].item())[1][0] * pos["quantity"].item()
            except KeyError:
                val = -np.inf
                # _log.error(f"Failed to get price for {inst} @ {tst}")

            # look if we need a different currency
            if pos.name[0] != base_currency:
                # we need to get an FX rate as well
                raise NotImplementedError("Multiple currencies not supported at the moment")

            return val

        res["liquidation_value"] = res.apply(mark_to_market, axis=1)
        res.index = pd.MultiIndex.from_arrays([res.index.get_level_values(-1), res.index.get_level_values(0)])
        return res

    return aggregator