#!/usr/bin/env bash

set -e
ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# replace "-{0.2.0}.zip" in all tox.ini files
TOX_FILES=$(find "$ABSOLUTE_PATH" -name tox.ini)
echo "$TOX_FILES"

# replace __version__ = '0.2.2' in all setup.py files
SETUP_FILES=$(find "$ABSOLUTE_PATH" -regex ".*/pandas-ml-[a-z\-]+/setup\.py")
echo "$SETUP_FILES"

# replace __version__ = '0.2.2' in all pandas_ml_*/__init__.py files
INIT_FILES=$(find "$ABSOLUTE_PATH" -regex ".*/pandas-ml-[a-z\-]+/pandas_ml_[a-z_]+/__init__\.py")
echo "$INIT_FILES"

