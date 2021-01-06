from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_common.decorator import MultiFrameDecorator


class TestMultiFrameDecorator(TestCase):

    def test_index(self):
        dfa = pd.DataFrame(np.random.random((10, 3)), columns=["a", "b", "c"])
        dfb = pd.DataFrame(np.random.random((8, 3)), columns=["A", "B", "C"])

        mdf1 = MultiFrameDecorator((dfa, dfb))
        mdf2 = MultiFrameDecorator((dfa, dfb), True)

        self.assertListEqual(list(range(10)), mdf1.index.to_list())
        self.assertListEqual(list(range(8)), mdf2.index.to_list())

    def test_loc(self):
        dfa = pd.DataFrame(np.random.random((10, 3)), columns=["a", "b", "c"])
        dfb = pd.DataFrame(np.random.random((8, 3)), columns=["A", "B", "C"])
        mdf = MultiFrameDecorator((dfa, dfb))

        self.assertEqual(10, len(mdf))
        self.assertEqual((3,), mdf.loc[3]._frames[0].shape)
        self.assertEqual((3,), mdf.loc[3]._frames[1].shape)
        pd.testing.assert_frame_equal(mdf.loc[3].as_joined_frame(), mdf.iloc[3].as_joined_frame())

    def test_joined(self):
        dfa = pd.DataFrame(np.random.random((10, 3)), columns=["a", "b", "c"])
        dfb = pd.DataFrame(np.random.random((8, 3)), columns=["A", "B", "C"])
        mdf = MultiFrameDecorator((dfa, dfb))

        joined_frame = mdf.as_joined_frame()
        self.assertNotIsInstance(joined_frame.index, pd.MultiIndex)
        self.assertEqual((10, 6), joined_frame.shape)

    def test_copy(self):
        dfa = pd.DataFrame(np.random.random((10, 3)), columns=["a", "b", "c"])
        dfb = pd.DataFrame(np.random.random((8, 3)), columns=["A", "B", "C"])
        mdf = MultiFrameDecorator((dfa, dfb))

        pd.testing.assert_frame_equal(mdf.as_joined_frame(), mdf.copy().as_joined_frame())
