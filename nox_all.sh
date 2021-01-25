#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# make sure we start with a clean environment
echo "clean environment"
rm -rf "$ABSOLUTE_PATH/.nox/*"

# run all tox tests
echo "run nox tests"
cd "$ABSOLUTE_PATH/pandas-ml-common" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-ml-utils" && nox -x && echo "done $(pwd)" && cd
cd "$ABSOLUTE_PATH/pandas-ml-utils-torch" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-ta-quant" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-ta-quant-plot" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-ml-quant" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-quant-data-provider" && nox -x && echo "done $(pwd)" && cd -
cd "$ABSOLUTE_PATH/pandas-ml-1ntegration-test" && nox -x && echo "done $(pwd)" && cd -

# TODO run private integration tests as nox tests as well
# TODO finalize remaining tox tests: pandas-quant-data-provider, pandas-ml-1ntegration-test, quant plot
#   cd "$ABSOLUTE_PATH/pandas-quant-data-provider" && tox && echo "done $(pwd)" && cd - || exit

# NOTE
# pandas-ml-airflow is in development
# pandas-ml-quant-rl is only experimental
# pandas-ml-keras is depricated
