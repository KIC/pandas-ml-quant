from typing import NamedTuple, Tuple, Callable, Union, List, Dict

from pandas_ml_common import Typing, naive_splitter


class FittingParameter(NamedTuple):
    splitter: Callable[[Typing.PdIndex], Tuple[Typing.PdIndex, Typing.PdIndex]] = naive_splitter()
    filter: Union['BaseCrossValidator', Tuple[int, Callable[[Typing.PatchedSeries], bool]]] = None
    cross_validation: Tuple[int, Callable[[Typing.PdIndex], Tuple[List[int], List[int]]]] = None
    epochs: int = 1
    batch_size: int = None
    fold_epochs: int = 1
    hyper_parameter_space: Dict = None
