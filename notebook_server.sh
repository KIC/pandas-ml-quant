#!/usr/bin/env bash

# add modules to path
for d in */ ; do
    PYTHONPATH="$PYTHONPATH:`pwd`/$d"
done

jupyter notebook --no-browser --port=8001 --ip 0.0.0.0 --NotebookApp.token=''
