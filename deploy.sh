#!/usr/bin/env bash

SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"
source "$SCRIPTPATH/.env.sh"

python setup.py sdist
twine upload dist/*

cp -r dist /tmp/
rm -rf *.egg-info dist

