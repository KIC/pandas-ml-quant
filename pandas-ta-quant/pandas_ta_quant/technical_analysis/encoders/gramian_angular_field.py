import logging

import numpy as _np
import pandas as _pd
from pyts.image import GramianAngularField as _GAF

from pandas_ml_common import Typing
from pandas_ml_common.utils import ReScaler
from pandas_ta_quant._decorators import *

_log = logging.getLogger(__name__)


# possible encoders
def pyts_gaf(time_steps, **kwargs):
    image_size = max(min(kwargs["image_size"] if "image_size" in kwargs else time_steps, time_steps), 1)

    def encoder(x):
        gaf = _GAF(**kwargs)
        return gaf.fit_transform(x)

    return encoder


def invertible_gaf_encoder(time_steps, **kwargs):
    rescale = True if "rescale" in kwargs and kwargs["rescale"] else False

    I = _np.ones(time_steps)

    def encoder(x):
        if rescale:
            x = ReScaler((_np.min(x), _np.max(x)), (0, 1), clip=True)(x)

        a = _np.sqrt(I - x ** 2)
        gaf = _np.outer(x, x) - _np.outer(a, a)
        return gaf

    return lambda x: _np.apply_along_axis(encoder, 1, x)


# technical analysis function
@for_each_top_level_row
def ta_gaf(df: Typing.PatchedPandas, columm_index_level=1, type='pyts', **kwargs):
    """
    :param df:
    :param columm_index_level:
    :param type:
    :param kwargs:
    :return: channel first 2D convolution GAF encoded matrix
    """

    if isinstance(df.columns, _pd.MultiIndex):
        # for each n'd level column create a dict like {Close: [(1, Close), (2, Close), ...]}
        l = columm_index_level
        columns = {c[l]: [c2 for c2 in df.columns.to_list() if c2[l] == c[l]] for c in df.columns.to_list()}
        res = _pd.DataFrame({}, index=df.index)

        # new we can loop feature for feature
        for feature, timesteps in columns.items():
            dff = df[timesteps]
            dff.columns = [t[:l] + t[l+1:] for t in timesteps]
            s = ta_gaf(dff, columm_index_level, type=type, **kwargs)
            res[f'{feature}_gaf'] = s

        return res
    else:
        time_steps = len(df.columns)

        if type == 'pyts':
            gaf_encoder = pyts_gaf(time_steps, **kwargs)
        elif type == 'invertible':
            if "rescale" not in kwargs or not kwargs["rescale"]:
                if _np.min(df.values) < 0 or _np.max(df.values) > 1:
                    _log.warning("data is not within [0, 1] and thus is not invertible!")

            gaf_encoder = invertible_gaf_encoder(time_steps, **kwargs)
        else:
            raise ValueError(f"unknown gaf encoder {type}! possible values ['pyts', 'invertible']")

        return _pd.Series(list(gaf_encoder(df.values)), name="gaf", index=df.index)


@for_each_top_level_row
@for_each_column
def ta_inverse_gasf(df: Typing.PatchedPandas):
    inv = np_inverse_gaf(df._.values)
    if inv.ndim == 2:
        return _pd.DataFrame(inv, index=df.index)
    else:
        if inv.shape[1] == 1:
            return _pd.DataFrame(inv[:,0,:], index=df.index)
        else:
            raise ValueError("should never get here ..")


def np_inverse_gaf(values):
    # values have shape channel, w, h
    if values.ndim == 4:
        return _np.array([np_inverse_gaf(values[sample]) for sample in range(values.shape[0])])
    else:
        return _np.array([_inverse_gaf(values[channel]) for channel in range(values.shape[0])])


def _inverse_gaf(values):
    return _np.sqrt(((_np.diag(values) + 1) / 2))
