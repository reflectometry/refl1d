"""
Generate Ni-film sample data files so that we can demonstrate file loading.
"""

from numpy.random import seed
from bumps.fitproblem import load_problem

from refl1d.names import SNS


def main():
    seed(1)
    problem = load_problem("nifilm-tof.py")
    for i, p in enumerate(problem.fitness.probe.probes):
        SNS.write_file("nifilm-tof-%d.dat" % (i + 1), p, title="Simulated 100 A Ni film")


if __name__ == "__main__":
    main()
