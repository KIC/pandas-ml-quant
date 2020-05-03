from unittest import TestCase

from pandas_ml_utils import FeaturesAndLabels


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

    def test_extraction(self):

        pass
