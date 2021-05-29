#!/bin/bash

set -e

# run tests
# ./nox_all.sh

# if all tests passed deploy all modules to pypi
./deploy.sh "pandas-ml-common" "pandas-ml-utils" "pandas-ml-utils-torch" "pandas-ta-quant" "pandas-ta-quant-plot"\
 "pandas-ml-quant" "pandas-quant-data-provider"


