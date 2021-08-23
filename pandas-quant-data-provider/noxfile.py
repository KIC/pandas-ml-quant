__version__ = '0.2.7'

import os
import shutil
from pathlib import Path

import nox

nox.options.envdir = '../.nox'


@nox.session(python=["3.8"], reuse_venv=False, venv_params=['--system-site-packages'])
def tests(session):
    dist_file = f"/tmp/pandas-quant-data-provider-{__version__}.zip"

    # create distribution and install
    session.install("-r", "requirements.txt", "-U")
    session.install("-r", "dev-requirements.txt")
    session.install(f"/tmp/pandas-ml-common-{__version__}.zip")

    session.run("python", "setup.py", "sdist",  "-d", "/tmp/", "--formats=zip", env={})
    session.install(dist_file)

    # create notebook kernels
    kernel = "data_provider"
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
    shutil.rmtree("pandas_quant_data_provider.egg-info")

    # make link check
    session.run("python", "pandas_quant_data_provider_test/check_links.py", dist_file)

    # freeze versions and rebuild with frozen versions
    session.run("python", "../freeze_versions.py", "requirements.txt", "dev-requirements.txt")
    session.run("python", "setup.py", "sdist",  "-d", "/tmp/", "--formats=zip", env={})