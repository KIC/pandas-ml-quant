#!/bin/bash

# add modules to path
for d in */ ; do
    PYTHONPATH="$PYTHONPATH:`pwd`/$d"
done

# run all tests
for d in */ ; do
    cd "$d"
    echo
    echo "$d"
    python -m unittest discover -s ./ -p "test_*.py" 2>&1 | grep -Eo 'FAILED.*|Ran.*'
    cd ..
done