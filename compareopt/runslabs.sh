#!/bin/bash
ROOT=$HOME/danse
REFL=$ROOT/refl1d
export PYTHONPATH=$ROOT/bumps:$ROOT/refl1d

for round in 1 2 3; do
  for num in 2 3 4 5 6; do 
    dest=T$num.$round
    echo "Running num=$num (round=$round) in $dest"
    $REFL/bin/refl1d_cli.py $REFL/compareopt/slabs.py --store=$dest $num --fit=dream --parallel --burn=9000  --steps=1000 --batch
    cp /tmp/problem $dest/initial_state
    grep chisq $dest/slabs.err
  done
done
