#!/bin/bash

# exit on any error
set -e

# add modules to path
SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"
source "$SCRIPTPATH/.env.sh"


# run all tests
for d in */ ; do
    cd "$d"
    echo
    echo "$d"
    python -m unittest discover -s ./ -p "test_*.py"
    cd ..
done