from unittest import TestCase

import pandas as pd

from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.data.reconstruction import map_prediction_to_target


class TestReconstruction(TestCase):

    def test_map_prediction_to_target_1o1(self):
        df = pd.DataFrame({
            (PREDICTION_COLUMN_NAME, "a"): [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (TARGET_COLUMN_NAME, "a"): [1, 1, 1],
            (TARGET_COLUMN_NAME, "b"): [2, 2, 2],
            (TARGET_COLUMN_NAME, "c"): [3, 3, 3],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        mapped = map_prediction_to_target(df, PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME)

        self.assertListEqual(mapped.index.to_list(),[(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)])
        self.assertListEqual(mapped[PREDICTION_COLUMN_NAME].to_list(), [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_map_prediction_to_target_open(self):
        df = pd.DataFrame({
            (PREDICTION_COLUMN_NAME, "a"): [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (TARGET_COLUMN_NAME, "a"): [2, 2, 2],
            (TARGET_COLUMN_NAME, "b"): [3, 4, 3]
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        mapped = map_prediction_to_target(df, PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME)

        self.assertListEqual(mapped.index.to_list(), [(0, (-float('inf'), 2)), (0, (2, 3)), (0, (3, float('inf'))), (1, (-float('inf'), 2)), (1, (2, 4)), (1, (4, float('inf'))), (2, (-float('inf'), 2)), (2, (2, 3)), (2, (3, float('inf')))])
        self.assertListEqual(mapped[PREDICTION_COLUMN_NAME].to_list(), [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_map_prediction_to_target_closed(self):
        df = pd.DataFrame({
            (PREDICTION_COLUMN_NAME, "a"): [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (TARGET_COLUMN_NAME, "a"): [1, 1, 1],
            (TARGET_COLUMN_NAME, "b"): [2, 2, 2],
            (TARGET_COLUMN_NAME, "c"): [3, 3, 3],
            (TARGET_COLUMN_NAME, "d"): [4, 4, 4],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        mapped = map_prediction_to_target(df, PREDICTION_COLUMN_NAME, TARGET_COLUMN_NAME)
        print(mapped.index.to_list())

        self.assertListEqual(mapped.index.to_list(), [(0, '(1, 2)'), (0, '(2, 3)'), (0, '(3, 4)'), (1, '(1, 2)'), (1, '(2, 3)'), (1, '(3, 4)'), (2, '(1, 2)'), (2, '(2, 3)'), (2, '(3, 4)')])
        self.assertListEqual(mapped[PREDICTION_COLUMN_NAME].to_list(), [1, 0, 0, 0, 1, 0, 0, 0, 1])