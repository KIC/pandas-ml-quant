#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# make sure we start with a clean environment
echo "clean environment"
rm -rf "$ABSOLUTE_PATH/.nox/*"

# run all tox tests
echo "run nox tests"
cd "$ABSOLUTE_PATH/pandas-ml-common" && nox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils" && nox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils-torch" && nox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ta-quant" && tox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ta-quant-plot" && tox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-quant" && tox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-quant-data-provider" && tox && echo "done $(pwd)" && cd - || exit
# FIXME pandas-quant-data-provider
# TODO run integration tess and private integration tests as nox tests as well
# TODO finalize remaining tox tests: pandas-quant-data-provider, pandas-ml-1ntegration-test, quant plot
#   cd "$ABSOLUTE_PATH/pandas-quant-data-provider" && tox && echo "done $(pwd)" && cd - || exit

# NOTE
# pandas-ml-airflow is in development
# pandas-ml-quant-rl is only experimental
# pandas-ml-keras is depricated
