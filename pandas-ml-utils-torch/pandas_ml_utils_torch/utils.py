from numbers import Number
from typing import Tuple, Union, Iterable, List, Set

import numpy as np
import torch as t

from pandas_ml_common import pd, MlTypes
from pandas_ml_common.utils import unpack_nested_arrays


def from_pandas(df: Union[MlTypes.PatchedDataFrame, List[MlTypes.PatchedDataFrame]], cuda: bool = False, default: t.Tensor = None) -> Union[t.Tensor, Tuple[t.Tensor, ...]]:
    if df is None and default is None:
        return None

    if isinstance(df, (List, Set, Tuple)):
        return tuple([from_pandas(f, cuda, default) for f in df])

    val = t.from_numpy(df.ML.get_values(split_multi_index_rows=False, squeeze=True)) if df is not None else default
    val = val.long() if is_df_int(df) else val.float()
    return val.cuda() if cuda else val.cpu()


def to_device(var, cuda):
    return var.cuda() if cuda else var.cpu()


def copy_weights(source_network, target_network):
    target_network.load_state_dict(source_network.state_dict())
    return target_network.train(source_network.training)


def wrap_applyable(func, nr_args=1, return_numpy=True):
    def wrapped(cells):
        x = []
        for i in range(nr_args):
            cell = (cells.iloc[i] if hasattr(cells, 'iloc') else cells[i]) if nr_args > 1 else cells
            if isinstance(cell, (np.ndarray, pd.Series, pd.DataFrame, Number, list, tuple, set)):
                if hasattr(cell, 'item') and sum(cell.shape) <= 1:
                    x.append(t.Tensor(unpack_nested_arrays(cell.item()) if cell.dtype == 'object' else [cell.item()]))
                elif hasattr(cell, 'values'):
                    x.append(t.Tensor(unpack_nested_arrays(cell.values) if cell.dtype == 'object' else cell.values))
                else:
                    x.append(t.Tensor(cell if isinstance(cell, Iterable) else [cell]))
            else:
                x.append(cell)

        x = func(*x)
        return (x.numpy() if sum(x.shape) > 1 else x.item()) if return_numpy else x

    return wrapped


def is_df_int(df):
    types = np.unique(df.dtypes.values)
    if len(types) == 1:
        if 'int' in str(types[0]):
            return True

    return False
