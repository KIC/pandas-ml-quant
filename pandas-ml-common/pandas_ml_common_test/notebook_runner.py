import nbformat
import os

from nbconvert.preprocessors import ExecutePreprocessor


def run_all_notebooks(notebooks_path, working_directory=None, kernel=os.getenv("TOX_KERNEL") or "python3", assert_nofail=True):
    notebooks = [os.path.join(notebooks_path, f) for f in os.listdir(notebooks_path) if f.endswith('.ipynb')]

    if working_directory is None:
        working_directory = notebooks_path

    results = {nb: run_notebook(nb, working_directory, kernel) for nb in notebooks if not "scratch" in nb}
    if assert_nofail:
        for file, (nb, err) in results.items():
            assert err == [], f'{file}: {err}'

    return results


def run_notebook(notebook_path, working_directory='/', kernel=os.getenv("TOX_KERNEL") or "python3"):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, iopub_timeout=10, kernel_name=kernel)
    proc.allow_errors = True

    proc.preprocess(nb, {'metadata': {'path': working_directory}})
    output_path = os.path.join(dirname, '{}.outnb'.format(nb_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb, f)

    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)

    return nb, errors

