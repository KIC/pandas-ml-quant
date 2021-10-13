from datetime import datetime
from functools import wraps
from string import Template
from typing import Union

from pandas_ml_common import MlTypes, FeaturesLabels
from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels, Extractor
from pandas_ml_common.utils import merge_kwargs
from pandas_ml_common.utils.time_utils import seconds_since_midnight
from pandas_ml_utils.df_patching.model_patch import DfModelPatch
from pandas_ml_utils.ml.fitting import Fit
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.model import Model
from pandas_ml_utils.ml.summary import Summary


class ModelContext(object):

    def __init__(self, df: MlTypes.PatchedDataFrame, file_name: str = None):
        self.df = df
        self.file_name = file_name

    @wraps(DfModelPatch.fit)
    def fit(self, *args, **kwargs) -> Fit:
        fit = self.df.model.fit(*args, **kwargs)

        if self.file_name is not None:
            file_name = Template(self.file_name).substitute({
                **kwargs,
                "V": f'{datetime.now().strftime("%Y%m-%d")}-{seconds_since_midnight()}'
            })

            fit.model.save(file_name)

        return fit

    def extract(self, model_or_fnl: Union[Model, FeaturesLabels], **kwargs) -> Extractor:
        if isinstance(model_or_fnl, Model):
            kwargs = merge_kwargs(model_or_fnl.kwargs, kwargs)
            return self.df.ML.extract(model_or_fnl.features_and_labels_definition, **kwargs)
        else:
            return self.df.ML.extract(model_or_fnl, **kwargs)

    @wraps(DfModelPatch.predict)
    def predict(self, *args, **kwargs) -> Union[MlTypes.PatchedDataFrame, Forecast]:
        return self.df.model.predict(Model.load(self.file_name), *args, **kwargs)

    @wraps(DfModelPatch.backtest)
    def backtest(self, *args, **kwargs) -> Summary:
        return self.df.model.backtest(Model.load(self.file_name), *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

            # eventually return True if all excetions are handled

