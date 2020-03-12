import contextlib
import unittest
import io
from pandas_ml_utils import pd


class TestFeatureSelection(unittest.TestCase):

    def test_feature_selection(self):
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5],
                           "featureB": [5, 4, 3, 2, 1],
                           "featureC": [1, 2, 1, 2, 1],
                           "labelA": [1, 2, 3, 4, 5],
                           "labelB": [5, 4, 3, 2, 1]})

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            df.model.feature_selection("labelA", lags=[2], show_plots=False)

        output = f.getvalue()

        # top features are A, B, C
        self.assertIn("Feature ranking:\n['labelB', 'featureA', 'featureB', 'featureC']", output)


if __name__ == '__main__':
    unittest.main()
