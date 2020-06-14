#!/usr/bin/env bash

# change environment
source .venv/bin/activate

# add development modules
SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"
source "$SCRIPTPATH/.env.sh"

# start server
jupyter notebook --no-browser --port=8001 --ip 0.0.0.0 --NotebookApp.token=''
