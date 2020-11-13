from unittest import TestCase

from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_quant import OneOverNModel
from pandas_ml_quant_test.config import DF_TEST_MULTI


class TestModel(TestCase):

    def test_one_over_n_model(self):
        df = DF_TEST_MULTI

        with df.model() as m:
            fit = m.fit(
                OneOverNModel(),
                training_data_splitter=duplicate_data()
            )

        print(fit)
