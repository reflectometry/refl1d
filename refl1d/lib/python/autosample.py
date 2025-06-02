from .reflectivity import reflectivity_amplitude
from .magnetic import calculate_u1_u3, magnetic_amplitude

import numpy as np
from numpy import int32, float64, complex128
from builtins import bool as boolean


def insert(arr, positions, values):
    # numba-accelerated version of numpy.insert
    ninput = len(arr)
    nvals = len(values)
    noutput = ninput + nvals
    target_positions = positions + np.arange(len(positions))
    output = np.empty((noutput,), dtype=arr.dtype)
    offsets = np.zeros((noutput,), dtype=boolean)
    offsets[target_positions] = True

    input_offset = 0
    val_index = 0
    for i in np.arange(noutput, dtype=np.int32):
        if offsets[i]:
            output[i] = values[val_index]
            val_index += 1
        else:
            output[i] = arr[input_offset]
            input_offset += 1
    return output


def get_refl_args(experiment):
    slabs = experiment._render_slabs()
    w = slabs.w
    rho, irho = slabs.rho, slabs.irho
    irho = abs(irho) + 1e-30
    sigma = slabs.sigma
    return w, rho, irho, sigma


def oversample_inplace(kz, dR, tol, w, rho, irho, sigma, max_sampling_iterations=100):
    new_kz = kz.copy()
    rho_index = np.zeros(kz.shape, int32)
    r = np.empty_like(kz, dtype=complex128)
    reflectivity_amplitude(w, sigma, rho, irho, -new_kz, rho_index, r)
    R = (r * np.conj(r)).real
    mapped_dR = dR.copy() / R
    out_of_tolerance = 1
    iterations = 0
    total_inserts = 0
    total_calc_R = 0
    out_of_tol = np.zeros((10), dtype=boolean)
    while out_of_tolerance > 0 and iterations < max_sampling_iterations:
        d12 = new_kz[:-2] - new_kz[1:-1]
        d23 = new_kz[1:-1] - new_kz[2:]
        d13 = new_kz[:-2] - new_kz[2:]

        y1 = R[:-2]
        y2 = R[1:-1]
        y3 = R[2:]

        # dy1 = mapped_dR[:-2]
        dy2 = mapped_dR[1:-1]
        # dy3 = mapped_dR[2:]

        common = (y1 * d23 - y2 * d13 + y3 * d12) / (6.0 * d13)
        a1, a2 = (d12 / d23 * common), (d23 / d12 * common)

        dx = new_kz[1:] - new_kz[:-1]
        # print(dict(a1=a1.shape, R=R.shape, d12=d12.shape, dx=dx.shape))

        rel_distance = np.zeros_like(dx, dtype=float64)
        # out_of_tol = np.zeros_like(dx, dtype=boolean)

        # print("max diff: ", (abs(a1) / (dy2 * y2)).max())
        diff1 = np.abs(a1) / (dy2 * y2)
        diff2 = np.abs(a2) / (dy2 * y2)
        # out2 = np.array(np.abs(a2) / (dy2 * y2) > tol, dtype=boolean)
        rel_distance[:-1] += diff1
        rel_distance[1:] += diff2
        out_of_tol = rel_distance > tol
        # out_of_tol[:-1] += out1
        # out_of_tol[1:] += out2

        # print(f"out_of_tol: {np.sum(out_of_tol)}\n")
        to_split = np.arange(len(dx), dtype=int32)[out_of_tol]
        new_kz_vals = new_kz[to_split] + dx[to_split] / 2.0
        new_dR = mapped_dR[to_split]

        new_r = np.empty_like(new_kz_vals, dtype=complex128)
        new_rho_index = np.zeros(new_kz_vals.shape, int32)
        reflectivity_amplitude(w, sigma, rho, irho, -new_kz_vals, new_rho_index, new_r)
        new_R = (new_r * np.conj(new_r)).real

        new_kz = insert(new_kz, to_split + 1, new_kz_vals)
        r = insert(r, to_split + 1, new_r)
        R = insert(R, to_split + 1, new_R)
        mapped_dR = insert(mapped_dR, np.clip(to_split - 1, 0, None), new_dR)

        iterations += 1
        out_of_tolerance = np.sum(out_of_tol)

    return new_kz, np.sum(out_of_tol), r, mapped_dR


def apply_autosampling(model, tolerance=0.05):
    w, rho, irho, sigma = get_refl_args(model)
    kz = model.probe.calc_Q / 2.0
    print("kz.shape", kz.shape)
    calc_q, calc_r = model.reflectivity()
    print(calc_r.shape, model.probe.dR.shape)
    # dR = model.probe.dR / calc_r
    # print(calc_r)
    # linear_kz = np.linspace(model.probe.Q.min() / 2, model.probe.Q.max() / 2, 100, endpoint=True)
    # linear_dR = np.interp(linear_kz, model.probe.Q / 2, np.abs(model.probe.dR / calc_r))
    new_kz, out_of_tol, r, dR = oversample_inplace(calc_q / 2.0, model.probe.dR, tolerance, w, rho, irho, sigma)
    # new_kz, out_of_tol, R, dR = oversample_inplace(linear_kz, linear_dR, tolerance, w, rho, irho, sigma)
    return new_kz, out_of_tol, r, dR


def autosampled_reflectivity_amplitude(depth, sigma, rho, irho, kz, rho_index, dR, tolerance=0.05):
    if dR is None:
        dR = np.full_like(kz, 0.01)

    if len(dR) != len(kz):
        raise ValueError("len(dR) != len(kz)")

    calc_kz, out_of_tol, r, dR = oversample_inplace(kz, dR, tolerance, depth, rho, irho, sigma)
    return calc_kz, r


def oversample_magnetic_inplace(kz, dR, tol, w, rho, irho, sigma, rhoM, thetaM, max_sampling_iterations=100):
    # dR is a list of relative errors in R for each cross-section present,
    # and will be None for missing cross-sections.

    # remove cross-sections that have dR == None

    new_kz = kz.copy()
    rho_index = np.zeros(kz.shape, int32)
    Ra = np.empty_like(kz, dtype=complex128)
    Rb = np.empty_like(kz, dtype=complex128)
    Rc = np.empty_like(kz, dtype=complex128)
    Rd = np.empty_like(kz, dtype=complex128)

    magnetic_amplitude(w, sigma, rho, irho, rhoM, u1, u3, -new_kz, rho_index, Ra, Rb, Rc, Rd)
    R = (r * np.conj(r)).real
    mapped_dR = dR.copy() / R
    out_of_tolerance = 1
    iterations = 0
    total_inserts = 0
    total_calc_R = 0
    out_of_tol = np.zeros((10), dtype=boolean)
    while out_of_tolerance > 0 and iterations < max_sampling_iterations:
        d12 = new_kz[:-2] - new_kz[1:-1]
        d23 = new_kz[1:-1] - new_kz[2:]
        d13 = new_kz[:-2] - new_kz[2:]

        y1 = R[:-2]
        y2 = R[1:-1]
        y3 = R[2:]

        # dy1 = mapped_dR[:-2]
        dy2 = mapped_dR[1:-1]
        # dy3 = mapped_dR[2:]

        common = (y1 * d23 - y2 * d13 + y3 * d12) / (6.0 * d13)
        a1, a2 = (d12 / d23 * common), (d23 / d12 * common)

        dx = new_kz[1:] - new_kz[:-1]
        # print(dict(a1=a1.shape, R=R.shape, d12=d12.shape, dx=dx.shape))

        rel_distance = np.zeros_like(dx, dtype=float64)
        # out_of_tol = np.zeros_like(dx, dtype=boolean)

        # print("max diff: ", (abs(a1) / (dy2 * y2)).max())
        diff1 = np.abs(a1) / (dy2 * y2)
        diff2 = np.abs(a2) / (dy2 * y2)
        # out2 = np.array(np.abs(a2) / (dy2 * y2) > tol, dtype=boolean)
        rel_distance[:-1] += diff1
        rel_distance[1:] += diff2
        out_of_tol = rel_distance > tol
        # out_of_tol[:-1] += out1
        # out_of_tol[1:] += out2

        # print(f"out_of_tol: {np.sum(out_of_tol)}\n")
        to_split = np.arange(len(dx), dtype=int32)[out_of_tol]
        new_kz_vals = new_kz[to_split] + dx[to_split] / 2.0
        new_dR = mapped_dR[to_split]

        new_r = np.empty_like(new_kz_vals, dtype=complex128)
        new_rho_index = np.zeros(new_kz_vals.shape, int32)
        reflectivity_amplitude(w, sigma, rho, irho, -new_kz_vals, new_rho_index, new_r)
        new_R = (new_r * np.conj(new_r)).real

        new_kz = insert(new_kz, to_split + 1, new_kz_vals)
        r = insert(r, to_split + 1, new_r)
        R = insert(R, to_split + 1, new_R)
        mapped_dR = insert(mapped_dR, np.clip(to_split - 1, 0, None), new_dR)

        iterations += 1
        out_of_tolerance = np.sum(out_of_tol)

    return new_kz, np.sum(out_of_tol), r, mapped_dR


def autosampled_magnetic_amplitude(depth, sigma, rho, irho, rhoM, thetaM, kz, rho_index, dR, tolerance=0.05):
    if dR is None:
        dR = np.full_like(kz, 0.01)

    if len(dR) != len(kz):
        raise ValueError("len(dR) != len(kz)")

    calc_kz, out_of_tol, r, dR = oversample_magnetic_inplace
    return calc_kz, r
