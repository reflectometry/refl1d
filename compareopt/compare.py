#!/usr/bin/env python
"""
Optimizer performance comparison
"""
import os
import sys
import Queue
import threading

root = 'out'

problems = [
    '../examples/xray/model.py', 
    ]

fitters = [
    ['--fit=de'],
#    ['--fit=rl'],
    ]

nthread = 8
nrepeat = 30

def run(queue, root):
    while True:
        i,j,pi,fj = queue.get()
        for k in range(repeat):
            dir = os.path.join(root, "P"+str(i), "F"+str(j), "T"+str(k))
            os.makedirs(dir)
            cmd = "../bin/refl1d --batch --stepmon --overwrite --random --store=%s '%s' %s"%(dir, pi, " ".join(fj))
            os.system(cmd)
        queue.task_done()

def main(problems, fitters, nthread, root):
    queue = Queue()
    for _ in range(nthread):
        t = Thread(target=run, args=(queue,root))
        t.daemon = True
        t.start()
    for i, pi in enumerate(problems):
        for j, fj in enumerate(fitters):
            queue.add((i,j,pi,fj))
    queue.join()
    
if __name__ == "__main__": 
    main(problems, fitters, nthread, root)
