#!/usr/bin/env bash

SCRIPT="$(readlink --canonicalize-existing "$0")"
SCRIPTPATH="$(dirname "$SCRIPT")"
source "$SCRIPTPATH/.env.sh"

deploy()
{
    echo "deploy: `pwd`"
    python setup.py sdist
    twine upload dist/*
    cp -r dist /tmp/
    rm -rf *.egg-info dist
}

if [ $# -eq 0 ]
  then
    deploy
else
  DIR=`pwd`
  for var in "$@"
    do
      cd "$DIR/$var"
      deploy
    done
fi

