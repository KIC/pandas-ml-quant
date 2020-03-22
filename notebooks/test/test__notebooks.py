import os
import unittest

from ._notebook_runner import run_notebook

PWD = os.path.dirname(os.path.abspath(__file__))


class TestNotebook(unittest.TestCase):

    def test_readme_nb(self):
        nb, errors = run_notebook(os.path.join(PWD, '..', 'Readme.ipynb'))
        self.assertEqual(errors, [])

    def test_plotting_nb(self):
        nb, errors = run_notebook(os.path.join(PWD, '..', 'plotting.ipynb'))
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
