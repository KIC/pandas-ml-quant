#!/usr/bin/env bash

if [ $# -ne 2 ]
  then
    echo "pass old_version new_version"
    exit -1
fi

OLD_VERSION=$1
NEW_VERSION=$2

set -e
for f in `fgrep $OLD_VERSION */* -d skip | sed -e's/\s*=\s*/:/g'`
do
  array=(${f//:/ })
  echo "sed -i -E \"s/(__version__)(\s*=\s*)(${array[2]})/\1\2'$NEW_VERSION'/\" ${array[0]}"
  sed -i -E "s/(__version__)(\s*=\s*)(${array[2]})/\1\2'$NEW_VERSION'/" ${array[0]}
done
