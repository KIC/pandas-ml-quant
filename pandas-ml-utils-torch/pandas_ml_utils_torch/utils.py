from typing import Tuple, Union

from pandas_ml_common import pd
import torch as t

from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.utils import unpack_nested_arrays


def from_pandas(df: pd.DataFrame, cuda: bool = False, default: t.Tensor = None) -> Union[t.Tensor, Tuple[t.Tensor, ...]]:
    if df is None and default is None:
        return None

    if isinstance(df, MultiFrameDecorator):
        return tuple([from_pandas(f, cuda, default) for f in df.frames()])

    val = (t.from_numpy(df._.get_values(split_multi_index_rows=False, squeeze=True)) if df is not None else default).float()
    return val.cuda() if cuda else val.cpu()


def to_device(var, cuda):
    return var.cuda() if cuda else var.cpu()


def copy_weights(source_network, target_network):
    target_network.load_state_dict(source_network.state_dict())
    return target_network.train(source_network.training)
