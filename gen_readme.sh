#!/usr/bin/env bash

# convert notebook
notedown notebooks/Readme.ipynb --to markdown > Readme.md

# fix path
sed -ie 's/](\.\.\//](/g' Readme.md

# add hint for nicely rendered output
ANCHOR='## Demonstration of all 3 libraries used together'
SUFFIX='-> Better browse the [notebooks\/Readme.ipynb](notebooks\/Readme.ipynb) with rendered outputs'
sed -ie "s/$ANCHOR/$ANCHOR $SUFFIX/g" Readme.md

# cleanup
rm -f Readme.mde
