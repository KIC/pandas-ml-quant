#!/bin/bash

#source .act/workaround_418.sh
sudo rm -rf .nox/*
sudo rm -rf ta-lib*
act -P ubuntu-latest=ubuntu-builder -b -j build

