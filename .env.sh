#!/usr/bin/env bash

SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"

# add modules to path
for d in $(find "$SCRIPTPATH" -maxdepth 1 -type d) ; do
    echo "add $d to python path"
    PYTHONPATH="$PYTHONPATH:$d"
done

echo "python path: $PYTHONPATH"
