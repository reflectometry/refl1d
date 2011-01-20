import sys; 
sys.path.insert(0,'../../..')
sys.path.insert(1,'../../../dream')

from refl1d.fitter import load_problem
from refl1d.snsdata import write_file

problem = load_problem('nifilm-tof.py')
for i,p in enumerate(problem.fitness.probe.probes):
    write_file('nifilm-tof-%d.dat'%(i+1), p, title="Simulated 100 A Ni film")
