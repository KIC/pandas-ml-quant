from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pandas_ml_common.sampling.sampler import Sampler, XYWeight
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common_test.config import TEST_MUTLI_INDEX_ROW_DF, TEST_DF


class TestSampler(TestCase):

    def test_simple_sample(self):
        sampler = Sampler(XYWeight(TEST_DF, None,  TEST_DF.tail()), splitter=None)
        samples = list(sampler.sample_for_training())

        self.assertEqual(1, len(samples))
        sample_one = samples[0]
        batches, test_sets = sample_one

        self.assertEqual(1, len(batches))
        self.assertEqual(0, len(test_sets))
        self.assertEqual(3, len(batches[0]))
        self.assertEqual(5, len(batches[0].x))
        self.assertEqual(5, len(batches[0].weight))
        self.assertIsNone(batches[0].y)
        self.assertIsNone(batches[0].y)

    def test_simple_sample_split(self):
        sampler = Sampler(TEST_DF, None, None, TEST_DF.tail(), splitter=lambda x, *args: (x[:3], x[3:]), epochs=2)
        samples = list(sampler.sample_cross_validation())
        row1 = samples[0]
        epoch, fold, train, test = row1

        self.assertEqual(2, len(samples))
        self.assertEqual(4, len(row1))
        self.assertEqual(-1, fold)
        self.assertEqual(4, len(train))
        self.assertEqual(4, len(test))
        self.assertEqual(3, len(train[0]))
        self.assertEqual(2, len(test[0]))
        self.assertIsNone(train[1])
        self.assertIsNone(test[2])

    def test_multiframe_decorator(self):
        sampler = Sampler(
            MultiFrameDecorator((TEST_DF.tail(), TEST_DF)),
            TEST_DF.tail(10),
            splitter=lambda x, *args: (x[:3], x[3:]),
            epochs=2
        )
        samples = list(sampler.sample_cross_validation())

        self.assertEqual(2, len(samples))
        self.assertEqual(2, len(samples[0][3][0]))

    def test_filter(self):
        sampler = Sampler(TEST_DF,
                          splitter=lambda x, *args: (x[:-3], x[-3:]),
                          training_samples_filter=(0, lambda r: r["Close"] > 300))
        samples = list(sampler.sample_cross_validation())

        self.assertEqual(101, len(samples[0][2][0]))

    def test_cross_validation(self):
        sampler = Sampler(TEST_DF, TEST_DF, TEST_DF.tail(30),
                          splitter=lambda x, *args: (x[:-3], x[-3:]),
                          cross_validation=(3, KFold(3).split),
                          epochs=2)

        samples = list(sampler.sample_cross_validation())

        self.assertListEqual([0, 1, 2, -1, 0, 1, 2, -1], [s[1] for s in samples])
        self.assertEqual(3, len(samples[3][3][0]))
        pd.testing.assert_frame_equal(samples[0][2][0], samples[4][2][0])
        pd.testing.assert_frame_equal(samples[1][2][0], samples[5][2][0])
        pd.testing.assert_frame_equal(samples[3][2][0], samples[7][2][0])

    def test_infinite_generator(self):
        sampler = Sampler(TEST_DF, None, None, TEST_DF.tail(), splitter=None, epochs=None)

        for i, s in enumerate(sampler.sample_cross_validation()):
            if i > 100:
                break

        self.assertGreater(i, 100)

    def test_infinite_generator_fail_cv(self):
        self.assertRaises(
            ValueError,
            lambda: Sampler(TEST_DF, None, None, TEST_DF.tail(30),
                            splitter=lambda x: (x[:3], x[3:]),
                            cross_validation=(3, KFold(3).split),
                            epochs=None)
      )

    def test_simple_sample_split_multiindex_row(self):
        sampler = Sampler(TEST_MUTLI_INDEX_ROW_DF, None, splitter=lambda x, *args: (x[:3], x[3:]), epochs=2)
        samples = list(sampler.sample_cross_validation())
        row1 = samples[0]
        epoch, fold, train, test = row1

        self.assertEqual(4, len(samples))
        self.assertEqual(4, len(row1))
        self.assertEqual(-1, fold)
        self.assertEqual(2, len(train))
        self.assertEqual(2, len(test))
        self.assertEqual(3, len(train[0]))
        self.assertEqual(2, len(test[0]))
        self.assertIsNone(train[1])

        # epochs are the most oter dimension, assert that we have all frames per epoch
        # and that the frames are equal in each epoch
        pd.testing.assert_frame_equal(samples[0][2][0], samples[2][2][0])
        pd.testing.assert_frame_equal(samples[1][2][0], samples[3][2][0])

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(samples[0][2][0].values, samples[1][2][0].values)

    def test_cross_validation_multiindex_row(self):
        sampler = Sampler(TEST_MUTLI_INDEX_ROW_DF, TEST_MUTLI_INDEX_ROW_DF,
                          splitter=lambda x, *args: (x[:-3], x[-3:]),
                          cross_validation=KFold(2),
                          epochs=2)

        samples = list(sampler.sample_cross_validation())

        self.assertListEqual([0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1], [s[1] for s in samples])
        pd.testing.assert_frame_equal(samples[0][2][0], samples[6][2][0])
        pd.testing.assert_frame_equal(samples[3][2][0], samples[9][2][0])

        for i in range(1, 6):
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(samples[0][2][0].values, samples[i][2][0].values)

    def test_full_epoch(self):
        sampler = Sampler(TEST_MUTLI_INDEX_ROW_DF, splitter=lambda x, *args: (x[:-3], x[-3:]), epochs=2)
        samples = list(sampler.sample_full_epochs())

        self.assertEqual(2, len(samples))
        pd.testing.assert_frame_equal(samples[0][1][0], pd.concat([TEST_MUTLI_INDEX_ROW_DF.loc[["A"]][:-3], TEST_MUTLI_INDEX_ROW_DF.loc[["B"]][:-3]], axis=0))
        pd.testing.assert_frame_equal(samples[0][1][0], samples[1][1][0])
        pd.testing.assert_frame_equal(samples[0][2][0], samples[1][2][0])
