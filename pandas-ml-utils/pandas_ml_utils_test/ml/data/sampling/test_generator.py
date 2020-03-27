import numpy as np
import pandas as pd
from unittest import TestCase

from pandas_ml_utils.ml.data.splitting import RandomSequences, DummySplitter
from pandas_ml_utils.ml.data.splitting.sampeling import DataGenerator, Sampler


class TestDataGenerator(TestCase):

    def test_generate_single_sample(self):
        data = pd.DataFrame({"a": np.arange(10)})
        test = (data, data, None)
        train = (data, data, None)
        gen = Sampler(train, test)

        samples = [s for s in gen.sample()]

        self.assertEqual(1, len(samples))

    def test_generate_multiple_sample(self):
        data = pd.DataFrame({"a": np.arange(10)})
        test = (data, data, None)
        train = (data, data, None)
        gen = Sampler(train, test, cross_validation=RandomSequences(0, 0.5, 12).cross_validation)

        samples = [s for s in gen.sample()]

        self.assertEqual(12, len(samples))

    def test_data_generator(self):
        pass

    def test_dummy_generator(self):
        features = pd.DataFrame({"a": np.arange(10)})
        targets = pd.DataFrame({"a": np.arange(10)})

        one = DataGenerator(DummySplitter(1), features, targets).complete_samples()
        two = DataGenerator(DummySplitter(2), features, targets).complete_samples()

        sampled1 = [s for s in one.sample()]
        sampled2 = [s for s in two.sample()]

        self.assertEqual(1, len(sampled1))
        self.assertEqual(2, len(sampled2))