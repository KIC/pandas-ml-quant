import os

from unittest import TestCase
from pandas_ml_utils import Model
from pandas_ml_common_test.notebook_runner import run_notebook
from pandas_ml_utils_test.config import DF_NOTES

PWD = os.path.dirname(os.path.abspath(__file__))


class TestModelContext(TestCase):

    def test_save_and_load_model_from_context(self):
        """given a clean workspace"""
        notebook_file = os.path.join(PWD, f'{__file__}.ipynb')
        out_notebook_file = os.path.join(PWD, f'{__file__}.py_all_output.ipynb')
        model_file = os.path.join(PWD, 'test_ctx.model')

        if os.path.exists(model_file): os.remove(model_file)
        if os.path.exists(out_notebook_file): os.remove(out_notebook_file)

        """when executing the notebook with a model context"""
        nb, errors = run_notebook(notebook_file, PWD)

        """then we have no errors and a saved model"""
        self.assertEqual(errors, [])
        self.assertTrue(os.path.exists(model_file))

        """and when we load the saved model"""
        model = Model.load(model_file)

        """then we can execute it"""
        self.assertEqual(2, len(DF_NOTES.model.predict(model, tail=2)))



