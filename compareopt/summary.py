import sys
import os
import glob
import numpy
import traceback

def main():
    assert len(sys.argv) == 2, "Need path to summarize"
    root = sys.argv[1]
    for problem in sorted(glob.glob(os.path.join(root,'*'))):
        if problem.endswith('pemu'): dof = 100-7
        elif problem.endswith('xray'): dof = 476-16
        for fitter in sorted(glob.glob(os.path.join(problem,'*'))):
            if fitter.endswith('dream'): continue
            if fitter.endswith('pt') or fitter.endswith('bfgs'):
                scale = 1
            else:
                scale = 2./dof
            for attempt in sorted(glob.glob(os.path.join(fitter,'*'))):
                filename = glob.glob(os.path.join(attempt,'*.log'))[0]
                try:
                    data = [[float(n) for n in s.split()]
                            for s in open(filename).readlines()
                            if s.strip() != '' and not s.startswith('#')]
                    result = numpy.array(data).T
                except KeyboardInterrupt:
                    raise
                except:
                    print traceback.print_exc()
                    print "while loading",filename
                else:
                    print attempt,min(result[1])*scale

if __name__ == "__main__": main()