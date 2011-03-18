#! /usr/bin/env python
"""
Optimizer performance comparison
"""
import os
import sys
import subprocess
from multiprocessing import Pool

root = 'out'

problems = {
    'xray': '../examples/xray/model.py',
    }

fitters = {
    'de': '--fit=de --steps=50',
    'rl': '--fit=rl --steps=50',
    }

nthread = None # Use #cpus
nrepeat = 3

def cmd(store,model,opts):
    pat = "nice ../bin/refl1d --batch --stepmon --overwrite --random --store=%s '%s' %s"
    return pat%(store,model,opts)

def run(job):
    modelname,model,fitname,opts,k = job
    store = os.path.join(root, modelname, fitname, "T%d"%k)
    if not os.path.exists(store):
        os.makedirs(store)
    print "starting %d %s %s\n"%(k,model,opts),
    subprocess.call(cmd(store,model,opts), shell=True)

def main(problems, fitters, nthread, root):
    if os.path.exists(root):
        raise RuntimeError("directory %s exists."%root)
    pool = Pool(processes=nthread)
    pool.map(run, [(i,pi,j,fj,k)
                   for i,pi in problems.items()
                   for j,fj in fitters.items()
                   for k in range(nrepeat)] )

if __name__ == "__main__":
    main(problems, fitters, nthread, root)
