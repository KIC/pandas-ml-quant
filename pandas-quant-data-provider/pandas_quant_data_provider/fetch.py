import logging
from typing import List, Union, Callable, Type, Dict

import numpy as np
import pandas as pd

from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_quant_data_provider.data_provider.yf import YahooSymbol
from pandas_quant_data_provider.symbol import Symbol

_log = logging.getLogger(__name__)


class QuantDataFetcher(object):

    def __init__(self, providers: Dict[Type, Symbol] = {}):
        self.provider_map = {
            YahooSymbol: YahooSymbol,
            str: YahooSymbol,
            **providers,
        }

        # copy string data type mapping for numpy string type, otherwise the user has to provide both all the time :-(
        self.provider_map[np.str_] = self.provider_map[str]

    def fetch_option_chain(self, symbol, max_maturities=None):
        return YahooSymbol(symbol).fetch_option_chain(symbol, max_maturities)

    def fetch_price_history(self,
                            *symbols: Union[Union[str, Symbol], List[Union[str, Symbol]]],
                            join: str = 'inner',
                            force_lowercase_header: bool = False,
                            **kwargs):
        symbols_array = self._init_shape(*symbols)
        frames = self._load(symbols_array, force_lowercase_header, **kwargs)
        row_frames = [pd.concat(row, join=join, axis=1) for row in frames]
        return row_frames[0] if len(row_frames) == 1 else pd.concat(row_frames, axis=0, join='outer')

    def _init_shape(self, *symbols) -> np.ndarray:
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

    def _load(self, symbols: np.ndarray, force_lowercase: bool, **kwargs):
        # also we want to control multi indexes i.e. we could pass [[AAPL, SPY], [ZM, QQQ]]
        frames = np.empty(symbols.shape, dtype=object)

        for i, row_symbols in enumerate(symbols):
            for j, symbol in enumerate(row_symbols):
                frames[i, j] = self._fetch_time_series(symbol, **kwargs)

                if force_lowercase:
                    frames[i, j].columns = [c.lower() for c in frames[i, j].columns]

                # fix MultiIndex if needed
                if frames.shape[0] > 1:
                    # we need a multi index index
                    if frames.shape[1] > 1:
                        # and we need a mutlti index column
                        frames[i, j].columns = pd.MultiIndex.from_product([[j], frames[i, j].columns])
                        frames[i, j].index = pd.MultiIndex.from_product([["/".join(row_symbols)], frames[i, j].index])
                    else:
                        frames[i, j].index = pd.MultiIndex.from_product([[symbol], frames[i, j].index])
                elif frames.shape[1] > 1:
                    # we need a multi index column
                    frames[i, j].columns = pd.MultiIndex.from_product([[symbol], frames[i, j].columns])

        return frames

    def _fetch_time_series(self, symbol, **kwargs) -> pd.DataFrame:
        symbol_implementation = symbol if isinstance(symbol, Symbol) else self.provider_map[type(symbol)](symbol)
        return call_callable_dynamic_args(symbol_implementation.fetch_price_history, **kwargs)
