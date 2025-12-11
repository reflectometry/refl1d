import sys
import os
import glob
import numpy
import traceback


def main():
    assert len(sys.argv) == 2, "Need path to summarize"
    root = sys.argv[1]
    for problem in sorted(glob.glob(os.path.join(root, "*"))):
        if problem.endswith("pemu"):
            dof = 100 - 7
        elif problem.endswith("xray"):
            dof = 476 - 16
        if True or not problem.endswith("pemu"):
            for fitter in sorted(glob.glob(os.path.join(problem, "*"))):
                # if fitter.endswith('dream'): continue
                if True or fitter.endswith("pt") or fitter.endswith("bfgs"):
                    scale = 1
                else:
                    scale = 2.0 / dof
                if True or not fitter.endswith("nm"):
                    for attempt in sorted(glob.glob(os.path.join(fitter, "*"))):
                        filelist = glob.glob(os.path.join(attempt, "*.log"))
                        filelist = [f for f in filelist if not f.endswith("e1085009.log")]
                        if len(filelist) > 1:
                            raise RuntimeException("Too many log files")
                        filename = filelist[0]
                        try:
                            # print("scanning %s"%filename)
                            data = [
                                [float(n) for n in s.split()]
                                for s in open(filename).readlines()
                                if s.strip() != "" and not s.startswith("#")
                            ]
                            result = numpy.array(data).T
                        except KeyboardInterrupt:
                            raise
                        except:
                            print(traceback.print_exc())
                            print("while loading %s" % filename)
                        else:
                            print("%s %s" % (attempt, min(result[1]) * scale))


if __name__ == "__main__":
    main()
