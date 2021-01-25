#!/bin/bash

# install act:
#   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
# then
#   create docker image and alias
#   see issue https://github.com/nektos/act/issues/418
#   docker build . -t ubuntu-builder
alias act="act -P ubuntu-latest=ubuntu-builder"