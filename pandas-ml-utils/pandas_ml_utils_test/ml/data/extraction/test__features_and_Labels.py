from functools import partial
from unittest import TestCase

from pandas_ml_utils import FeaturesAndLabels
from test.config import DF_TEST


class TestExtrationOfFeaturesAndLabels(TestCase):

    def test_repr(self):
        fnl = FeaturesAndLabels(
            features=[1, "A", lambda x: x * 2],
            labels=[1, "A", lambda x: x * 2],
        )

        # print(repr(fnl).replace('"', "\\\""))
        self.assertEqual(
            "FeaturesAndLabels(	[1, 'A', '            features=[1, \"A\", lambda x: x * 2],\\n'], 	[1, 'A', '            labels=[1, \"A\", lambda x: x * 2],\\n'], 	None, 	None, 	None, 	None, 	{})",
            repr(fnl)
        )

    def test__extract_label_comprehension(self):
        df = DF_TEST.copy()

        (features, _), labels, targets, weights, gross_loss = df._.extract(
            FeaturesAndLabels(
                features=["Close"],
                labels=[partial(lambda i, df: df["Close"].rename(f"Close_{i}"), i) for i in range(5, 10)]
            )
        )

        #print(labels.tail())
        self.assertListEqual(
            ["Close_5", "Close_6", "Close_7", "Close_8", "Close_9"],
            labels.columns.tolist()
        )

    def test_extraction(self):

        pass
