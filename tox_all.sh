#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# make sure we start with a clean environment
echo "clean environment"
rm -rf "$ABSOLUTE_PATH/.tox"

# run all tox tests
echo "run tox tests"
cd "$ABSOLUTE_PATH/pandas-ml-common" && tox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils" && tox && echo "done $(pwd)" && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils-torch" && tox && echo "done $(pwd)" && cd - || exit
# cd "$ABSOLUTE_PATH/pandas-ml-utils-keras" && tox && echo "done $(pwd)" && cd - || exit      # temporarily deprecate keras
cd "$ABSOLUTE_PATH/pandas-ml-quant" && tox && echo "done $(pwd)" && cd - || exit
# TODO finalize remaining tox tests: pandas-ml-quant-data-provider, pandas-ml-1ntegration-test
#   cd "$ABSOLUTE_PATH/pandas-ml-quant-data-provider" && tox && echo "done $(pwd)" && cd - || exit
# TODO finalize remaining tox tests: pandas-ml-quant-data-provider, pandas-ml-1ntegration-test
#   cd "$ABSOLUTE_PATH/pandas-ml-quant-plot" && tox && echo "done $(pwd)" && cd - || exit
# NOTE
# pandas-ml-airflow is in development
# pandas-ml-quant-rl is only experimental