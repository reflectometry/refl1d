#!/bin/bash
ROOT=$HOME/src
REFL=$ROOT/refl1d
#export PYTHONPATH=$ROOT/bumps:$ROOT/refl1d
PROG=refl1d

for round in 1 2 3; do
  for num in 4 5 6; do
    dest=T$num.$round
    echo "Running num=$num (round=$round) in $dest"
    mkdir $dest
    $PROG $REFL/compareopt/slabs.py --store=$dest $num `pwd`/$dest/initial_state --fit=dream --parallel --burn=9000  --steps=1000 --batch
    grep chisq $dest/slabs.err
  done
done
