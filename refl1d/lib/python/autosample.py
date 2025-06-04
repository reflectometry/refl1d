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
    mapped_dR = dR.copy()
    out_of_tolerance = 1
    iterations = 0
    total_inserts = 0
    total_calc_R = 0
    out_of_tol = np.zeros((10), dtype=boolean)
    while out_of_tolerance > 0 and iterations < max_sampling_iterations:
        d12 = new_kz[:-2] - new_kz[1:-1]  # delta between first and second points
        d23 = new_kz[1:-1] - new_kz[2:]  # delta between second and third points
        d13 = new_kz[:-2] - new_kz[2:]  # delta between first and third points

        y1 = R[:-2]  # first point reflectivity
        y2 = R[1:-1]  # second point reflectivity
        y3 = R[2:]  # third point reflectivity

        # dy1 = mapped_dR[:-2]
        dy2 = mapped_dR[1:-1]  # second point mapped dR
        # dy3 = mapped_dR[2:]

        # Calculate the common term for the cubic interpolation
        common = (y1 * d23 - y2 * d13 + y3 * d12) / (6.0 * d13)
        # Calculate the coefficients for the cubic interpolation
        a1 = d12 / d23 * common
        # a1 is the average difference between the quadratic interpolation
        # between the first and second points, (also going through the third point)
        # and the linear interpolation between the first and second points
        a2 = d23 / d12 * common
        # a2 is the average difference between the quadratic interpolation
        # between the second and third points, (also going through the first point)
        # and the linear interpolation between the second and third points

        dx = new_kz[1:] - new_kz[:-1]
        # print(dict(a1=a1.shape, R=R.shape, d12=d12.shape, dx=dx.shape))

        rel_distance = np.zeros_like(dx, dtype=float64)
        # out_of_tol = np.zeros_like(dx, dtype=boolean)

        # print("max diff: ", (abs(a1) / (dy2 * y2)).max())
        # diff is the difference scaled by the acceptable error in the reflectivity
        diff1 = np.abs(a1) / dy2
        diff2 = np.abs(a2) / dy2
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
        mapped_dR = insert(mapped_dR, to_split, new_dR)

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


def oversample_magnetic_inplace(
    kz, dRa, dRb, dRc, dRd, tol, w, rho, irho, sigma, sld_b, u1, u3, max_sampling_iterations=100
):
    # dR is a list of relative errors in R for each cross-section present,
    # and will be None for missing cross-sections.

    # remove cross-sections that have dR == None

    new_kz = kz.copy()
    rho_index = np.zeros_like(kz, dtype=int32)
    ra = np.empty_like(kz, dtype=complex128)
    rb = np.empty_like(kz, dtype=complex128)
    rc = np.empty_like(kz, dtype=complex128)
    rd = np.empty_like(kz, dtype=complex128)

    magnetic_amplitude(w, sigma, rho, irho, sld_b, u1, u3, -new_kz, rho_index, ra, rb, rc, rd)
    Ra = (ra * np.conj(ra)).real
    Rb = (rb * np.conj(rb)).real
    Rc = (rc * np.conj(rc)).real
    Rd = (rd * np.conj(rd)).real

    mapped_dRa = dRa.copy() if dRa is not None else np.zeros(kz.shape, dtype=float64)
    mapped_dRb = dRb.copy() if dRb is not None else np.zeros(kz.shape, dtype=float64)
    mapped_dRc = dRc.copy() if dRc is not None else np.zeros(kz.shape, dtype=float64)
    mapped_dRd = dRd.copy() if dRd is not None else np.zeros(kz.shape, dtype=float64)

    iterations = 0
    total_inserts = 0
    total_calc_R = 0
    out_of_tol = np.zeros((10), dtype=boolean)
    for xsi in range(1):
        R = [Ra, Rb, Rc, Rd][xsi]
        mapped_dR = [mapped_dRa, mapped_dRb, mapped_dRc, mapped_dRd][xsi]
        if [dRa, dRb, dRc, dRd][xsi] is None:
            continue
        out_of_tolerance = 1
        # print(f"Starting oversampling for cross-section {xsi} with tolerance {tol}")
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
            diff1 = np.abs(a1) / dy2
            diff2 = np.abs(a2) / dy2
            # out2 = np.array(np.abs(a2) / (dy2 * y2) > tol, dtype=boolean)
            rel_distance[:-1] += diff1
            rel_distance[1:] += diff2
            out_of_tol = rel_distance > tol
            # out_of_tol[:-1] += out1
            # out_of_tol[1:] += out2

            # print(f"out_of_tol: {np.sum(out_of_tol)}\n")
            to_split = np.arange(len(dx), dtype=int32)[out_of_tol]
            new_kz_vals = new_kz[to_split] + dx[to_split] / 2.0
            new_dRa = mapped_dRa[to_split]
            new_dRb = mapped_dRb[to_split]
            new_dRc = mapped_dRc[to_split]
            new_dRd = mapped_dRd[to_split]
            # new_dR = mapped_dR[to_split]

            # new_r = np.empty_like(new_kz_vals, dtype=complex128)
            new_ra = np.empty_like(new_kz_vals, dtype=complex128)
            new_rb = np.empty_like(new_kz_vals, dtype=complex128)
            new_rc = np.empty_like(new_kz_vals, dtype=complex128)
            new_rd = np.empty_like(new_kz_vals, dtype=complex128)

            magnetic_amplitude(
                w, sigma, rho, irho, sld_b, u1, u3, -new_kz_vals, rho_index, new_ra, new_rb, new_rc, new_rd
            )
            # reflectivity_amplitude(w, sigma, rho, irho, -new_kz_vals, new_rho_index, new_r)
            new_Ra = (new_ra * np.conj(new_ra)).real
            new_Rb = (new_rb * np.conj(new_rb)).real
            new_Rc = (new_rc * np.conj(new_rc)).real
            new_Rd = (new_rd * np.conj(new_rd)).real

            # new_R = (new_r * np.conj(new_r)).real

            new_kz = insert(new_kz, to_split + 1, new_kz_vals)

            ra = insert(ra, to_split + 1, new_ra)
            rb = insert(rb, to_split + 1, new_rb)
            rc = insert(rc, to_split + 1, new_rc)
            rd = insert(rd, to_split + 1, new_rd)

            Ra = insert(Ra, to_split + 1, new_Ra)
            Rb = insert(Rb, to_split + 1, new_Rb)
            Rc = insert(Rc, to_split + 1, new_Rc)
            Rd = insert(Rd, to_split + 1, new_Rd)

            R = [Ra, Rb, Rc, Rd][xsi]

            mapped_dRa = insert(mapped_dRa, to_split, new_dRa) if mapped_dRa is not None else None
            mapped_dRb = insert(mapped_dRb, to_split, new_dRb) if mapped_dRb is not None else None
            mapped_dRc = insert(mapped_dRc, to_split, new_dRc) if mapped_dRc is not None else None
            mapped_dRd = insert(mapped_dRd, to_split, new_dRd) if mapped_dRd is not None else None

            mapped_dR = [mapped_dRa, mapped_dRb, mapped_dRc, mapped_dRd][xsi]

            # mapped_dR = insert(mapped_dR, np.clip(to_split - 1, 0, None), new_dR)

            iterations += 1
            out_of_tolerance = np.sum(out_of_tol)
            # print(f"out_of_tol: {np.sum(out_of_tol)}\n")

    return new_kz, np.sum(out_of_tol), ra, rb, rc, rd, mapped_dRa, mapped_dRb, mapped_dRc, mapped_dRd


def autosampled_magnetic_amplitude(
    depth, sigma, rho, irho, sld_b, u1, u3, kz, rho_index, dRa, dRb, dRc, dRd, tolerance=0.05
):
    if dRa is None and dRb is None and dRc is None and dRd is None:
        dRa = np.full_like(kz, 0.01)
        dRb = np.full_like(kz, 0.01)
        dRc = np.full_like(kz, 0.01)
        dRd = np.full_like(kz, 0.01)

    if any([len(dR) != len(kz) for dR in [dRa, dRb, dRc, dRd] if dR is not None]):
        raise ValueError(f"len(dR) != len(kz)({len(kz)})", [len(dR) for dR in [dRa, dRb, dRc, dRd] if dR is not None])

    calc_kz, out_of_tol, ra, rb, rc, rd, dRa, dRb, dRc, dRd = oversample_magnetic_inplace(
        kz, dRa, dRb, dRc, dRd, tolerance, depth, rho, irho, sigma, sld_b, u1, u3
    )
    return calc_kz, ra, rb, rc, rd
