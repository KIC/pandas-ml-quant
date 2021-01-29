import os
from unittest import TestCase

from pandas_ml_common_test.notebook_runner import run_all_notebooks

PWD = os.path.dirname(os.path.abspath(__file__))


class TestNotebooks(TestCase):

    def test_all_notebooks(self):
        notebooks_path = os.path.join(PWD, '..', 'examples')
        run_all_notebooks(notebooks_path, assert_nofail=True)
