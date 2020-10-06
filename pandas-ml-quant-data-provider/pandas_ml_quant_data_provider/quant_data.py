import logging
import os
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from simple_plugin_loader import Loader
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

_log = logging.getLogger(__name__)


class DataProvider(object):

    symbols_table_name = 'symbols'
    time_series_table_name = 'timeseries'

    @staticmethod
    def today():
        return datetime.today()

    @staticmethod
    def last_business_day():
        days = max(DataProvider.today().weekday() * -1 + 7, 3)
        return DataProvider.today() + timedelta(days=days)

    @staticmethod
    def tomorrow():
        return DataProvider.today() + timedelta(days=1)

    def __init__(self, file, *args, **kwargs):
        self.db_name = type(self).__name__.lower()
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(file)), self.db_name)
        self.conn_str = f'sqlite:///{self.db_path}/{self.db_name}.db'

        _log.info(f"connecting: {self.conn_str}")
        self.engine: Engine = create_engine(self.conn_str, echo=True)

    @abstractmethod
    def update_symbols(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def update_quotes(self, **kwargs):
        raise NotImplemented

    @abstractmethod
    def has_symbol(self, symbol: str, **kwargs):
        # returns true if this symbol is supported by this data provider
        raise NotImplemented

    @abstractmethod
    def load(self, symbol: str, **kwargs):
        # load the symbol from the cache
        # append the newest data to the cace
        # return data as DataFrame
        raise NotImplemented


class QuantData(object):

    def __init__(self,
                 plugin_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugins'),
                 plugins: Dict[str, type] = None
                 ):
        self.plugin_path = plugin_path
        self.loader = Loader()
        self.plugins: Dict[str, DataProvider] = plugins if plugins else \
            {k: v() for k, v in self.loader.load_plugins(self.plugin_path, DataProvider, True).items()}

    def update_database(self):
        for plugin in self.plugins.values():
            plugin.update_symbols()

    def load(self,
             *symbols: Union[str, List[str]],
             join: str = 'inner',
             provider_splitter: str = '|',
             force_provider: Union[str, List[str]] = None):
        symbols_array = self._init_shape(*symbols)
        frames = self._load(symbols_array, provider_splitter=provider_splitter, force_provider=force_provider)
        row_frames = [pd.concat(row, join=join, axis=1) for row in frames]
        return row_frames[0] if len(row_frames) == 1 else pd.concat(row_frames, axis=0, join='outer')

    def _init_shape(self, *symbols: [Union[str], List[str]]) -> np.ndarray:
        # fix shape
        if len(symbols) == 1:
            # check if load(["ABC", "DEF"]) or load([["ABC", "DEF"]])
            if isinstance(symbols[0], List) and len(symbols[0]) > 0:
                if isinstance(symbols[0][0], List):
                    # load([["ABC", "DEF"]])
                    symbols = np.array(symbols[0])

                    # magic
                    if symbols.shape[0] == 1:
                        symbols = symbols.reshape(-1, 1)
                else:
                    # load(["ABC", "DEF"])
                    symbols = np.array([symbols[0]])
            else:
                # load("ABC"
                symbols = np.array([symbols])
        elif len(symbols) > 1:
            # check if load("ABC", "DEF") or load(["ABC", "DEF"], ["XYZ", "123"])
            if isinstance(symbols[0], List) and len(symbols[0]) > 0:
                # load(["ABC", "DEF"], ["XYZ", "123"])
                symbols = np.array(symbols)
            else:
                # load("ABC", "DEF")
                symbols = np.array([symbols])
        else:
            raise ValueError("No symbols provided")

        return symbols

    def _load(self, symbols: np.ndarray, provider_splitter='|', force_provider: Union[str, List[str]] = None):
        # loop all providers and if the symbol is supported return data from this provder
        # if multiple providers support this data raise ambiguity exception
        # also we want to control multi indexex i.e. we could pass [[AAPL, SPY], [ZM, QQQ]]
        frames = np.empty(symbols.shape, dtype=object)

        for i, row_symbols in enumerate(symbols):
            for j, column_symbol in enumerate(row_symbols):
                symbol_provider = column_symbol.split(provider_splitter)
                frames[i, j] = self._fetch_time_series(*symbol_provider, force_provider=force_provider)

                # check relevance
                if frames[i, j].index[-1] < DataProvider.last_business_day():
                    _log.warning(f"last data point of {symbol_provider} is older then yesterday: {frames[i, j].index[-1]}")

                # fix indices
                if frames.shape[0] > 1:
                    # we need a multi index index
                    if frames.shape[1] > 1:
                        # and we need a mutlti index column
                        frames[i, j].columns = pd.MultiIndex.from_product([[j], frames[i, j].columns])
                        frames[i, j].index = pd.MultiIndex.from_product([["/".join(row_symbols)], frames[i, j].index])
                    else:
                        frames[i, j].index = pd.MultiIndex.from_product([[column_symbol], frames[i, j].index])
                elif frames.shape[1] > 1:
                    # we need a multi index column
                    frames[i, j].columns = pd.MultiIndex.from_product([[column_symbol], frames[i, j].columns])

        return frames

    def _fetch_time_series(self, symbol, provider=None, force_provider=None):
        providers = self._get_provider(symbol, force_provider or provider)

        for p in providers:
            try:
                df = p.load(symbol)
                if df is not None:
                    return df
            except Exception as e:
                _log.warning(f"{p} failed to download {symbol}: {e}")

        # if none of the providers succeeded we finally throw an exception
        raise ValueError(f"failed downloading {symbol} using providers: {providers}")

    def _get_provider(self, symbol, provider: Union[List[str], str] = None) -> List[DataProvider]:
        if provider is None:
            providers = [p for name, p in self.plugins.items() if p.has_symbol(symbol) and (provider is None or isinstance(p, provider))]
        else:
            providers = [self.plugins[p] for p in provider] if isinstance(provider, List) else [self.plugins[provider]]

        if len(providers) <= 0:
            raise ValueError(f"No data provider for {symbol} in {self.plugins}")

        if len(providers) > 1:
            _log.warning(f"Ambiguous symbol has multiple providers {providers} use first one {providers[0]}\n"
                         f"To override the provider pass the symbol with a dedicated provider i.e.' {symbol}|yahoo'")

        return providers

