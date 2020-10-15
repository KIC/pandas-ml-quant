from datetime import datetime
from string import Template
from typing import Callable, Tuple, Dict, Union, List

from pandas_ml_common import Typing
from pandas_ml_common.sampling import naive_splitter
from pandas_ml_common.utils.time_utils import seconds_since_midnight
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels
from pandas_ml_utils.ml.model.base_model import Model


class ModelContext(object):

    def __init__(self, df: Typing.PatchedDataFrame, file_name: str = None):
        self.df = df
        self.file_name = Template(file_name).substitute({
            "V": f'{datetime.now().strftime("%Y%m-%d")}-{seconds_since_midnight()}'
        }) if file_name is not None else None

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

    def extract(self, model: Union[Model, FeaturesAndLabels]) -> FeaturesWithLabels:
        return self.df._.extract(model.features_and_labels if isinstance(model, Model) else model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

            # eventually return True if all excetions are handled
