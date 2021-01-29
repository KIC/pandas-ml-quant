__version__ = '0.2.1'

import os
import shutil
from pathlib import Path

import nox

nox.options.envdir = '../.nox'


@nox.session(python=["3.8"], reuse_venv=False, venv_params=['--system-site-packages'])
def tests(session):
    # install testing requirements and local dependencies
    session.install("-r", "requirements.txt")
    session.install(f"/tmp/pandas-ml-common-{__version__}.zip")
    session.install(f"/tmp/pandas-ml-utils-{__version__}.zip")
    session.install(f"/tmp/pandas-ta-quant-{__version__}.zip")
    session.install(f"/tmp/pandas-ml-quant-{__version__}.zip")
    session.install(f"/tmp/pandas-quant-data-provider-{__version__}.zip")

    # create notebook kernels
    kernel = "ml_integrationtest"
    session.run("python", "-m", "ipykernel", "install", "--user", "--name", kernel, "--display-name", f"{kernel} py38")

    # run tests
    session.run("python", "-m", "unittest", "discover", env={"TOX_KERNEL": kernel})

    # clean up kernels
    kernels_home_dir = os.path.join(Path.home(), ".local", "share", "jupyter", "kernels", kernel)
    if os.path.exists(kernels_home_dir):
        print(f"clean up dir {kernels_home_dir}")

        try:
            shutil.rmtree(kernels_home_dir)
        except Exception as e:
            print(e)
