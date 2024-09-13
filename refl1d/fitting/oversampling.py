import argparse

import numpy as np


def get_optimal_single_oversampling(model, tolerance=0.05, max_oversampling=201, seed=1, verbose=False):
    """
    Determine how much oversampling is required to adequately calculate resolution smearing

    Algorithm:
      - the model is oversampled to a very high number (max_oversampling)
      - smeared R is calculated at max_oversampling and stored as reference "ideal" R
      - oversampling is iterated (starting at 1), and smeared R is calculated and compared to "ideal" R
      - points are "within tolerance" if (R(Q) - R_ref(Q)) / dR(Q) < tolerance
      - the value of oversampling at which a point becomes within tolerance is recorded for that point
      - when all points are within tolerance the loop ends and the function reports the recommended oversampling

    Args:
        model (refl1d.experiment.Experiment): an Experiment, containing probe and sample
        tolerance (float, optional): allowed deviation of R from ideal R (multiplied by dR). Defaults to 0.05
        max_oversampling (int, optional): A very high oversampling that is expected to exceed
            the requirements for support (used to generate reference R). Defaults to 201.

    Returns:
        oversampling (int): suggested oversampling to get all R within tolerance
        optimal_oversampling: for each probe, per-Q array of oversampling needed to get R(Q) within tolerance
    """
    # get a list of probes, which will have length one for unpolarized:
    probes = model.probe.xs if hasattr(model.probe, "xs") else [model.probe]
    # sample = model.sample

    # initialize the per-Q recommended oversampling to max_oversampling
    optimal_oversampling = [None if p is None else np.ones_like(p.dR, dtype=int) * max_oversampling for p in probes]
    Q = []
    model.probe.oversample(max_oversampling, seed=seed)
    model._cache = {}
    R_ref = model.reflectivity()
    if not isinstance(R_ref, list):
        R_ref = [R_ref]

    max_diff = np.inf
    oversampling = 0
    while max_diff > tolerance and oversampling < max_oversampling:
        oversampling += 1
        if verbose:
            print("trying oversampling = {:d}".format(oversampling), end="\r")
        model.probe.oversample(oversampling)
        model._cache = {}
        R = model.reflectivity()
        if not isinstance(R, list):
            R = [R]
        current_max_diff = 0
        for oos, p, r, r_ref in zip(optimal_oversampling, probes, R, R_ref):
            if p is None:
                Q.append(None)
                continue

            Q.append(p.Q)
            diff = np.abs(r[1] - r_ref[1]) / p.dR
            current_max_diff = max(current_max_diff, diff.max())

            # if an R value is now within tolerance, and the optimal_oversampling has not been set
            # (which is indicated by the fact that it is still set to max_oversampling), then
            # set it to the current oversampling:
            to_set = np.logical_and(diff < tolerance, oos == max_oversampling)
            oos[to_set] = oversampling

            # if an R value is now out of tolerance, make sure that the optimal_oversampling is reset
            # to the "unset" value of max_oversampling
            to_reset = diff > tolerance
            oos[to_reset] = max_oversampling

        max_diff = current_max_diff  # to be checked against tolerance on the next iteration

    # reset model cache again before leaving
    model._cache = {}
    if verbose:
        print("Recommended oversampling: {:d}".format(oversampling))

    return oversampling, optimal_oversampling, Q


def analyze_fitproblem(problem, tolerance=0.05, max_oversampling=201, seed=1, plot=False):
    if plot:
        from matplotlib import pyplot as plt
    models = problem.models if hasattr(problem, "models") else [problem]
    oversampling = []
    local_oversampling = []
    for i_model, model in enumerate(models):
        print("model: {:d}".format(i_model))
        oversampling_i = []
        local_oversampling_i = []
        oversampling.append(oversampling_i)
        local_oversampling.append(local_oversampling_i)
        parts = model.parts if hasattr(model, "parts") else [model]
        for i_part, part in enumerate(parts):
            # print(part)
            oversampling_ii, local_oversampling_ii, Q = get_optimal_single_oversampling(
                part, tolerance, max_oversampling, seed=seed
            )
            print("\tpart: {:d}, oversampling: {:d}".format(i_part, oversampling_ii))
            oversampling_i.append(oversampling_ii)
            local_oversampling_i.append(local_oversampling_ii)
            if plot:
                plt.figure()
                for i_oos, (oos, qq) in enumerate(zip(local_oversampling_ii, Q)):
                    if oos is not None:
                        plt.plot(qq, oos, label="xs: {:d}".format(i_oos))
                plt.title("model: {:d}, part: {:d}".format(i_model, i_part))
                plt.xlabel("Q (inv. A)")
                plt.ylabel("required oversampling")
                plt.legend()

        # there is one probe instance shared between parts in MixedExperiment: use
        # largest recommended oversampling.
        parts[0].probe.oversample(max(oversampling_i))

    if plot:
        plt.show()

    return oversampling, local_oversampling


def main():
    import sys
    from bumps.cli import load_model, load_best

    parser = argparse.ArgumentParser(
        description="""
        Algorithm:
      - the model is oversampled to a very high number (max_oversampling)
      - smeared R is calculated at max_oversampling and stored as reference "ideal" R
      - oversampling is iterated (starting at 1), and smeared R is calculated and compared to "ideal" R
      - points are "within tolerance" if (R(Q) - R_ref(Q)) / dR(Q) < tolerance
      - the value of oversampling at which a point becomes within tolerance is recorded for that point
      - when all points are within tolerance the loop ends and the function reports the recommended oversampling
      """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot optimal oversampling for each probe (default = False)"
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolerance for expression: (R - R_ideal)/dR < tolerance (default=0.05)",
    )
    parser.add_argument(
        "-m",
        "--max_oversampling",
        type=int,
        default=201,
        help="Max oversampling (also used to calculate R_ideal; default=201)",
    )
    parser.add_argument("-p", "--pars", type=str, default="", help="retrieve starting point from .par file")

    parser.add_argument("modelfile", type=str, nargs=1, help="refl1d model file")
    parser.add_argument("modelopts", type=str, nargs="*", help="options passed to the model")

    opts = parser.parse_args(None if sys.argv[1:] else ["-h"])

    problem = load_model(opts.modelfile[0], model_options=opts.modelopts)
    if opts.pars:
        load_best(problem, opts.pars)

    analyze_fitproblem(problem, opts.tolerance, opts.max_oversampling, opts.plot)


if __name__ == "__main__":
    main()
