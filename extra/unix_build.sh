#!/bin/sh

set -x

export PYTHONPATH=../periodictable:../bumps
python setup.py build
python test.py
python check_examples.py --chisq
(cd doc && make html pdf)
cp doc/_build/latex/Refl1D.pdf .
