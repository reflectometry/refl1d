#! /usr/bin/env python
"""
Optimizer performance comparison
"""
import os
import sys
import subprocess
from multiprocessing import Pool


problems = {
    'xray': '../examples/xray/model.py',
    'pemu': '../doc/examples/superlattice/PEMU-web.py',
    }

# Aim for evals; assume problem size is 10
root = 'out2'
evals = 1000000
nrepeat = 5
fitters = {
    'de'   : '--fit=de     --steps=%d --pop=10'%(evals//100),
    'ps'   : '--fit=ps     --steps=%d --pop=10'%(evals//100),
    'rl1'  : '--fit=rl     --steps=%d --pop=10 --starts=1'%(evals//100),
    'rln'  : '--fit=rl     --steps=%d --starts=100 --pop=0.5'%(evals//500),
    'pt'   : '--fit=pt     --steps=%d --pop=10 --burn=0'%(evals//100),
    #'dream': '--fit=dream  --steps=2000 --pop=10 --burn=8000',
    'bfgs' : '--fit=newton --steps=100 --starts=%d'%(evals//1000),
    'nm'   : '--fit=amoeba --steps=1000 --starts=%d'%(evals//1000),
    }

nthread = None # Use #cpus

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
