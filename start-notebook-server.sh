#!/usr/bin/env bash

SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"
source "$SCRIPTPATH/.env.sh"

jupyter notebook --no-browser --port=8001 --ip 0.0.0.0 --NotebookApp.token=''
