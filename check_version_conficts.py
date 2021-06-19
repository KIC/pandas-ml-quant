
import subprocess
import sys
import re

DEFAULT_REQUIREMENTS = [
    #"pandas-ml-1ntegration-test-private/requirements.txt",
    #"pandas-ml-1ntegration-test/requirements.txt",
    #"pandas-ml-airflow/requirements.txt",
    "pandas-ml-common/requirements.txt",
    "pandas-ml-quant/requirements.txt",
    #"pandas-ml-quant-rl/requirements.txt",
    "pandas-ml-utils/requirements.txt",
    #"pandas-ml-utils-tf/requirements.txt",
    "pandas-ml-utils-torch/requirements.txt",
    "pandas-quant-data-provider/requirements.txt",
    "pandas-ta-quant-plot/requirements.txt",
    "pandas-ta-quant/requirements.txt",
    #"streamlit_apps/requirements.txt",
]


def get_frozen_filename(filename):
    parts = filename.split('.')
    parts = parts[:-1] + ['frozen'] + parts[-1:]
    return '.'.join(parts)


if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_REQUIREMENTS
    dependencies = [tuple(dep.replace('\n', '').split('==')) for requirements_file in files for dep in open(get_frozen_filename(requirements_file)).readlines()]

    for i, (d, v) in enumerate(dependencies):
        versions = [v2 for j, (d2, v2) in enumerate(dependencies) if j != i and d2 == d]
        if len(set(versions)) > 1:
            print(f"ERROR: version conflict: {d}: {set(versions)}")
