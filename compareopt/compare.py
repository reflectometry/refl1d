#! /usr/bin/env python
"""
Optimizer performance comparison
"""
import os
import sys
import subprocess
from multiprocessing import Pool

root = 'out2'

problems = {
    'xray': '../examples/xray/model.py',
    'pemu': '../doc/examples/superlattice/PEMU-web.py',
    }

# Aim for 1000000 fcalls
fitters = {
    'de'   : '--fit=de     --steps=10000 --pop=10',
    'ps'   : '--fit=ps     --steps=10000 --pop=10',
    'rl1'  : '--fit=rl     --steps=10000 --pop=10 --starts=1',
    'rln'  : '--fit=rl     --steps=1000 --starts=100 --pop=0.5',
    'pt'   : '--fit=pt     --steps=100000 --pop=10 --burn=0',
    #'dream': '--fit=dream  --steps=2000 --pop=10 --burn=8000',
    'bfgs' : '--fit=newton --steps=100 --starts=10000',
    'nm'   : '--fit=amoeba --steps=1000 --starts=1000',
    }

nthread = None # Use #cpus
nrepeat = 20

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
        raise RuntimeError("directory <%s> exists."%root)
    pool = Pool(processes=nthread)
    pool.map(run, [(i,pi,j,fj,k)
                   for k in range(nrepeat)
                   for i,pi in problems.items()
                   for j,fj in fitters.items()
                   ] )

if __name__ == "__main__":
    main(problems, fitters, nthread, root)
