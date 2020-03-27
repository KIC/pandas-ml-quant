import numpy as np
import pandas as pd
from unittest import TestCase

from pandas_ml_utils.ml.data.splitting import RandomSequences
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