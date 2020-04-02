from typing import Callable

import numpy as np
import pandas as pd


PRICE = "price"
ASSET_VALUE = "asset_value"
POS_OPEN = "position_open"
POS_CLOSE = "position_close"
POS_NET = "position_net"
CASH_SLIPP = "slippage"
CASH_OPEN = "cash_open"
CASH_CLOSE = "cash_close"
CASH_NET = "cash_net"
NET = "net"


class TransactionLog(object):
    def __init__(self):
        super().__init__()
        self.transactions_open = {}
        self.transactions_close = {}

    def add_open_transaction(self, iloc: int, amount: float):
        amount = float(amount)

        if iloc in self.transactions_open:
            self.transactions_open[iloc] = [*self.transactions_open[iloc], amount]
        else:
            self.transactions_open[iloc] = [amount]

    def add_close_transaction(self, iloc: int, amount: float):
        amount = float(amount)

        if iloc in self.transactions_close:
            self.transactions_close[iloc] = [*self.transactions_close[iloc], amount]
        else:
            self.transactions_close[iloc] = [amount]

    def evaluate(self, prices: pd.Series, slippage: Callable[[float], float] = lambda value: 0):
        with pd.option_context('mode.chained_assignment', None):
            df = prices.to_frame()
            open_keys = [*self.transactions_open.keys()]
            close_keys = [*self.transactions_close.keys()]

            # calculate positions
            df[POS_OPEN] = 0
            df[POS_OPEN].iloc[open_keys] = [np.array([v]).sum() for v in self.transactions_open.values()]

            df[POS_CLOSE] = 0
            df[POS_CLOSE].iloc[close_keys] = [np.array([v]).sum() for v in self.transactions_close.values()]

            df[POS_NET] = (df[POS_OPEN] + df[POS_CLOSE]).cumsum()

            # calculate cash balance
            df[CASH_OPEN] = 0
            df[CASH_CLOSE] = 0
            df[CASH_SLIPP] = 0

            open = df[POS_OPEN].iloc[open_keys] * prices.iloc[open_keys]
            df[CASH_OPEN].iloc[open_keys] -= open

            close = df[POS_CLOSE].iloc[close_keys] * -prices.iloc[close_keys]
            df[CASH_CLOSE].iloc[close_keys] += close

            df[CASH_SLIPP].iloc[open_keys] += open.apply(slippage)
            df[CASH_SLIPP].iloc[close_keys] += close.apply(slippage)

            df[CASH_NET] = (df[CASH_OPEN] + df[CASH_CLOSE] + df[CASH_SLIPP]).cumsum()
            pd.set_option('display.max_columns', 500)

            # calculate position value
            df[ASSET_VALUE] = df[POS_NET] * prices
            df[NET] = df[CASH_NET] + df[ASSET_VALUE]

            return df


class StreamingTransactionLog(object):
    OPEN = 1
    CLOSE = -1

    def __init__(self):
        self.log = TransactionLog()
        self.current_index = 0
        self.current_position = None

    def rebalance(self, balance):
        if self.current_position is None:
            self.log.add_open_transaction(self.current_index, balance)
            self.current_position = 0
        else:
            if balance == 0:
                if self.current_position != 0:
                    self.log.add_close_transaction(self.current_index, -self.current_position)
            elif self.current_position > 0 and balance < 0:
                # swing long to short
                self.log.add_close_transaction(self.current_index, -self.current_position)
                self.log.add_open_transaction(self.current_index, balance)
            elif self.current_position < 0 and balance > 0:
                # swing short to long
                self.log.add_close_transaction(self.current_index, -self.current_position)
                self.log.add_open_transaction(self.current_index, balance)
            else:
                if balance > 0:
                    # long
                    delta = balance - self.current_position
                    if delta > 0:
                        self.log.add_open_transaction(self.current_index, delta)
                    else:
                        self.log.add_close_transaction(self.current_index, delta)
                else:
                    # short
                    delta = balance - self.current_position
                    if delta > 0:
                        self.log.add_close_transaction(self.current_index, delta)
                    else:
                        self.log.add_open_transaction(self.current_index, delta)

        self.current_index += 1
        self.current_position = balance

    def perform_action(self, action: int, amount):
        action = int(action)
        amount = float(amount)

        if action > 0:
            self.log.add_open_transaction(self.current_index, amount)
        elif action < 0:
            self.log.add_close_transaction(self.current_index, action)

        self.current_index += 1
        return self.current_index

    def evaluate(self, prices: pd.Series, slippage: Callable[[float], float] = lambda value: 0):
        return self.log.evaluate(prices, slippage)
