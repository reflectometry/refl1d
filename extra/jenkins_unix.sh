#!/bin/sh

# should be testing python 2.6, 2.7, 3.3, 3.4
PYTHON=${PYTHON:-python}
export PYTHON
set -x

# Pull the latest bumps/periodictable from github, rather than pypi.
# A true bleeding edge build would also install the latest versions of
# the various dependent packages (numpy, scipy, numdifftools, pyparsing,
# etc.), but that is a project for a different day.
$PYTHON -m pip install --no-deps -t external https://github.com/bumps/bumps/tarball/master
$PYTHON -m pip install --no-deps -t external https://github.com/pkienzle/periodictable/tarball/master

PYTHONPATH=$WORKSPACE/external:$PYTHONPATH
export PYTHONPATH

$PYTHON setup.py build
$PYTHON test.py

# check that the docs build
$PYTHON check_examples.py --chisq
(cd doc && make html pdf)
cp doc/_build/latex/Refl1D.pdf .
