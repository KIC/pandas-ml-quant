#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# make sure we start with a clean environment
echo "clean environment"
sudo rm -rf "$ABSOLUTE_PATH/.nox/*"

run_tests () {
  wdir="$ABSOLUTE_PATH/$1"
  echo ----------------------------------------------------------------------------------------------------------
  echo $wdir
  echo ----------------------------------------------------------------------------------------------------------
  echo

  cd "$wdir"
  nox -x
  echo "done $(pwd)"
  cd -
}

# run all tox tests
echo "run nox tests"
run_tests "pandas-ml-common"
run_tests "pandas-ml-utils"
run_tests "pandas-ml-utils-torch"
run_tests "pandas-ta-quant"
run_tests "pandas-ta-quant-plot"
run_tests "pandas-ml-quant"
run_tests "pandas-quant-data-provider"
run_tests "pandas-ml-1ntegration-test"

# run spellchecker
pyspelling

# TODO run private integration tests as nox tests as well
# TODO finalize remaining tox tests: pandas-quant-data-provider, pandas-ml-1ntegration-test, quant plot
#   cd "$ABSOLUTE_PATH/pandas-quant-data-provider" && tox && echo "done $(pwd)" && cd - || exit

# NOTE
# pandas-ml-airflow is in development
# pandas-ml-quant-rl is only experimental
# pandas-ml-keras is depricated
