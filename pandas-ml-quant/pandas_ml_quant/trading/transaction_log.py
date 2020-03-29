from typing import Callable

import pandas as pd


class TransactionLog(object):
    OPEN = 1
    CLOSE = -1
    SKIP = 0

    PRICE = "price"
    ASSET_VALUE = "asset value"
    POSITION = "position"
    CASH = "cash"
    NET = "net"

    def __init__(self):
        super().__init__()
        self.transactions = []
        self.index = []
        self.index_open = []
        self.index_close = []
        self.current_index = 0

    def action(self, action: int, amount):
        if action > 0:
            self.add_open_transaction(self.current_index, amount)
        elif action < 0:
            if len(self.transactions) > 0 and self.transactions[-1] > 0:
                self.add_close_transaction(self.current_index, amount)

        self.current_index += 1
        return self.current_index

    def add_open_transaction(self, iloc: int, amount: float):
        self.transactions.append((self.transactions[-1] if len(self.transactions) > 0 else 0) + amount)
        self.index_open.append(iloc)
        self.index.append(iloc)

    def add_close_transaction(self, iloc: int, amount: float):
        assert self.transactions[-1] != 0, "can not close non existing position"

        # make sure we close one bar later as we can see in this toy example:
        # price | position | value
        # 10    | 1        | 10
        # 20    | 0        | 20       # we sold here so we need to have a value
        # 15    | 0        | 0

        self.transactions.append((self.transactions[-1] if len(self.transactions) > 0 else 0) + amount)
        self.index_close.append(iloc + 1)
        self.index.append(iloc + 1)

    def evaluate(self, prices: pd.Series, slippage: Callable[[float], float] = lambda value: 0):
        df = prices.to_frame()
        df.columns = [TransactionLog.PRICE]
        df[TransactionLog.POSITION] = None
        df[TransactionLog.CASH] = 0

        # set positions
        if len(self.index) > 0:
            df[TransactionLog.POSITION].iloc[self.index] = self.transactions

        # evaulate asset value
        df = df.fillna(method='ffill', axis=0).fillna(0)
        df[TransactionLog.ASSET_VALUE] = df[TransactionLog.PRICE] * df[TransactionLog.POSITION]

        # add cash effects
        open = df[TransactionLog.ASSET_VALUE].iloc[self.index_open]
        df[TransactionLog.CASH].iloc[self.index_open] -= open - open.apply(slippage)

        close = df[TransactionLog.ASSET_VALUE].shift(1).iloc[self.index_close]
        df[TransactionLog.CASH].iloc[self.index_close] += close + close.apply(slippage)

        # cum sum cash balance
        df[TransactionLog.CASH] = df[TransactionLog.CASH].fillna(0).cumsum()

        # calculate net value
        df[TransactionLog.NET] = df[TransactionLog.CASH] + df[TransactionLog.ASSET_VALUE]
        return df

