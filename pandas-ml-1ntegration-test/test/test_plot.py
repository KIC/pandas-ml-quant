from unittest import TestCase
import pandas_ml_quant
from pandas_ml_quant_test.config import DF_TEST

print(pandas_ml_quant.__version__)


class TestPlot(TestCase):

    def test_line(self):
        df = DF_TEST

        qp = df.q.ta_plot(rows=1)
        qp.line(df.q.ta_adx())
