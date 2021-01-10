#!/bin/bash

# see issue https://github.com/nektos/act/issues/418
# docker build . -t ubuntu-builder
alias act="act -P ubuntu-latest=ubuntu-builder"