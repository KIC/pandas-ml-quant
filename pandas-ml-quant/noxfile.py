__version__ = '0.2.4'

import os
import shutil
from pathlib import Path

import nox

nox.options.envdir = '../.nox'


@nox.session(python=["3.8"], reuse_venv=False, venv_params=['--system-site-packages'])
def tests(session):
    dist_file = f"/tmp/pandas-ml-quant-{__version__}.zip"

    # install testing requirements and local dependencies
    session.install("-r", "dev-requirements.txt")
    session.install("-r", "requirements.txt")
    session.install(f"/tmp/pandas-ml-common-{__version__}.zip")
    session.install(f"/tmp/pandas-ml-utils-{__version__}.zip")
    session.install(f"/tmp/pandas-ta-quant-{__version__}.zip")

    # create distribution and install
    session.run("python", "setup.py", "sdist",  "-d", "/tmp/", "--formats=zip", env={})
    session.install(dist_file, "--no-deps")

    # create notebook kernels
    kernel = "ml_quant"
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

    # clean up egg info
    shutil.rmtree("pandas_ml_quant.egg-info")

    # make link check
    session.run("python", "pandas_ml_quant_test/check_links.py", dist_file)

    # freeze versions and rebuild with frozen versions
    session.run("python", "../freeze_versions.py", "requirements.txt", "dev-requirements.txt")
    session.run("python", "setup.py", "sdist",  "-d", "/tmp/", "--formats=zip", env={})