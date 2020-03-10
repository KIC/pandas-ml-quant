import pandas as pd

import pandas_ml_quant.indicators.multi_object as multi_object_indicators
import pandas_ml_quant.indicators.single_object as single_object_indicators
import pandas_ml_quant.indicators.auto_regression as auto_regressive_indicators


class Quant(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # add all indicators
        for indicator_functions in [single_object_indicators, multi_object_indicators, auto_regressive_indicators]:
            for indicator_function in dir(indicator_functions):
                if indicator_function.startswith("ta_"):
                    setattr(self, indicator_function, getattr(indicator_functions, indicator_function))


    @property
    def plot(self):
        pass


