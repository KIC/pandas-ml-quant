import os
import unittest
from unittest import TestCase

from pandas_ml_common_test.notebook_runner import run_notebook
from pandas_ml_utils import Model
from pandas_ml_utils_test.config import DF_NOTES
import subprocess
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
BIN = sys.executable
MODEL_FILE = os.path.join(PWD, 'test_ctx.model')
CLEAN_SCRIPT = os.path.join(PWD, "subprocess_load_module.py")


def test_model_import(*args):
    with subprocess.Popen([BIN, CLEAN_SCRIPT, *args], stdout=subprocess.PIPE) as p:
        return p.stdout.read().decode('utf-8')


class TestModelContext(TestCase):

    @unittest.skip("only used for debugging")
    def test_subprocess(self):
        out = test_model_import(MODEL_FILE)
        print(out)

        self.assertIn(MODEL_FILE, out)
        self.assertIn("MLPRegressor", out)
        self.assertIn("authentic", out)

    def test_save_and_load_model_from_context(self):
        """given a clean workspace"""
        notebook_file = os.path.join(PWD, f'{__file__}.ipynb')
        out_notebook_file = os.path.join(PWD, f'{__file__}.py_all_output.ipynb')

        if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
        if os.path.exists(out_notebook_file): os.remove(out_notebook_file)

        """when executing the notebook with a model context"""
        nb, errors = run_notebook(notebook_file, PWD, kernel=os.getenv("TOX_KERNEL") or "python3")

        """then we have no errors and a saved model"""
        self.assertEqual(errors, [])
        self.assertTrue(os.path.exists(MODEL_FILE))

        """and when we load the saved model"""
        model = Model.load(MODEL_FILE)

        """then we can execute it"""
        self.assertEqual(2, len(DF_NOTES.model.predict(model, tail=2)))

        """and we can load the odel in a clean name space"""
        out = test_model_import(MODEL_FILE)
        print(out)

        self.assertIn(MODEL_FILE, out)
        self.assertIn("authentic", out)





