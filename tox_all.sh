#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# make sure we start with a clean environment
rm -rf "$ABSOLUTE_PATH/.tox"

# run all tox tests
cd "$ABSOLUTE_PATH/pandas-ml-common" && tox && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils" && tox && cd - || exit
cd "$ABSOLUTE_PATH/pandas-ml-utils-torch" && tox && cd - || exit
# cd "$ABSOLUTE_PATH/pandas-ml-utils-keras" && tox && cd - || exit      # temporarily deprecate keras
cd "$ABSOLUTE_PATH/pandas-ml-quant" && tox && cd - || exit
# TODO finalize remaining tox tests: pandas-ml-quant-data-provider, pandas-ml-1ntegration-test

# NOTE
# pandas-ml-airflow is in development
# pandas-ml-quant-rl is only experimental