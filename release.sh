#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# run all tox tests
bash "$ABSOLUTE_PATH/tox_all.sh"

# TODO generate readme ..

# source release: push all files, create tag, push tags
git push
tag=$(grep __version__ "$ABSOLUTE_PATH/pandas-ml-utils/setup.py" | grep \'.*\' -o)
git tag $tag
git push --tags

# upload files to a release (pip install github-binary-upload)
github-binary-upload -l KIC/pandas-ml-quant pandas-ml-quant-data-provider/pandas_ml_quant_data_provider/plugins/investing/investing.db
github-binary-upload -l KIC/pandas-ml-quant pandas-ml-quant-data-provider/pandas_ml_quant_data_provider/plugins/yahoo/yahoo.db

# release to pypi
deploy()
{
    cd "$1" || exit
    echo "deploy: `pwd`"
    python setup.py sdist
    twine upload dist/*
    cp -r dist /tmp/
    rm -rf *.egg-info dist
    cd - || exit
}

deploy pandas-ml-common
deploy pandas-ml-utils
# TODO and the rest

# increase version
# TODO ...