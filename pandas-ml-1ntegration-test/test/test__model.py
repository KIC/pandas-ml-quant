from unittest import TestCase

import pandas_ml_quant
from pandas_ml_utils import FeaturesAndLabels, Model
from test.config import DF_TEST

print(pandas_ml_quant.__version__)


class TestModel(TestCase):

    def test_simple_model(self):
        df = DF_TEST.copy()

        # TODO
        df.model.fit(Model(FeaturesAndLabels([], []))) # FIXME implement me

        pass
