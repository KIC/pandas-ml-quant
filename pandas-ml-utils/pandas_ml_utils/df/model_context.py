from collections import namedtuple
from typing import Callable, Tuple, Dict, Union, List

from pandas_ml_common import Typing
from pandas_ml_common.sampling import naive_splitter
from pandas_ml_utils.ml.model.base_model import Model
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels


class ModelContext(object):

    def __init__(self, df: Typing.PatchedDataFrame, file_name: str = None):
        self.df = df
        self.file_name = file_name

    def fit(self,
            model: Callable[[], Model],
            training_data_splitter: Callable[[Typing.PdIndex], Tuple[Typing.PdIndex, Typing.PdIndex]] = naive_splitter(),
            training_samples_filter: Union['BaseCrossValidator', Tuple[int, Callable[[Typing.PatchedSeries], bool]]] = None,
            cross_validation: Tuple[int, Callable[[Typing.PdIndex], Tuple[List[int], List[int]]]] = None,
            hyper_parameter_space: Dict = None,
            **kwargs
            ):

        fit = self.df.model.fit(
            model, training_data_splitter, training_samples_filter, cross_validation, hyper_parameter_space, **kwargs)

        if self.file_name is not None:
            fit.model.save(self.file_name)

        return fit

    def extract(self, model: Model) -> FeaturesWithLabels:
        return self.df._.extract(model.features_and_labels)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

            # eventually return True if all excetions are handled

