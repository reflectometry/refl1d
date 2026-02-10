def test():
    import numpy
    from refl1d.names import Material, Experiment, NeutronProbe, FitProblem, silicon, air

    nickel = Material("Ni")

    # This defines a chemical formula, Ni, for which the program knows the
    # density in advance since it has densities for all elements.  By using
    # chemical composition, we can compute scattering length densities for
    # both X-ray and neutron beams from the same sample description.
    # Alternatively, we could take a more traditional approach and define
    # nickel as a specific SLD for our beam

    # nickel = SLD(rho=9.4)

    # The '#' character on the above line means that line is a comment, and
    # it won't be evaluated.

    # With our materials defined (silicon, nickel and air), we can combine
    # them into a sample. The substrate will be silicon with a 5 |Ang|
    # 1-\ $\sigma$ Si:Ni interface.  The nickel layer is 100 |Ang| thick
    # with a 5 |Ang| Ni:Air interface.  Air is on the surface.

    sample = silicon(0, 5) | nickel(100, 5) | air

    # Our sample definition is complete, so now we need to specify the
    # range of values we are going to view.  We will use the
    # `numpy <http://numpy.scipy.org/>`_ library, which extends python
    # with vector and matrix operations.  The `linspace` function below
    # returns values from 0 to 5 in 100 steps for incident angles
    # from 0\ |deg| to 5\ |deg|.

    T = numpy.linspace(0, 5, 30)

    # From the range of reflection angles, we can create a neutron probe. The probe
    # defines the wavelengths and angles which are used for the measurement as well
    # as their uncertainties.  From this the resolution of each point can be
    # calculated.  We use constants for angular divergence ``dT=0.01``\ |deg|,
    # wavelength ``L=4.75`` |Ang| and wavelength dispersion ``dL=0.0475`` in this
    # example, but each angle and wavelength is independent.

    probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

    # Combine the neutron probe with the sample stack to define an
    # experiment.  Using chemical formula and mass density, the same
    # sample can be simulated for both neutron and x-ray experiments.

    M = Experiment(probe=probe, sample=sample)

    # Generate a random data set with 5% noise. While not necessary
    # to display a reflectivity curve, it is useful in showing how
    # the data set should look.

    M.simulate_data(5)

    # Combine a set of experiments into a fitting problem.  The problem
    # is used by refl1d for all operations on the model.

    assert numpy.allclose(
        probe.calc_Q,
        numpy.array(
            [
                0.0,
                0.00796095,
                0.01592183,
                0.02388257,
                0.03184308,
                0.03980332,
                0.04776319,
                0.05572262,
                0.06368156,
                0.07163991,
                0.07959762,
                0.08755461,
                0.0955108,
                0.10346613,
                0.11142052,
                0.11937391,
                0.12732621,
                0.13527736,
                0.14322728,
                0.15117591,
                0.15912317,
                0.16706899,
                0.17501329,
                0.18295601,
                0.19089707,
                0.19883641,
                0.20677394,
                0.2147096,
                0.22264332,
                0.23057502,
            ]
        ),
    )

    probe.oversample(3, seed=1)

    assert numpy.allclose(
        probe.calc_Q,
        numpy.array(
            [
                -0.00027897,
                0.0,
                0.00075572,
                0.00748785,
                0.0076777,
                0.00796095,
                0.01487038,
                0.01592183,
                0.01624207,
                0.02317883,
                0.02388257,
                0.02441192,
                0.03130617,
                0.03184308,
                0.0321773,
                0.03904917,
                0.03980332,
                0.04107146,
                0.04717257,
                0.04753826,
                0.04776319,
                0.05572262,
                0.05606903,
                0.05635448,
                0.06275672,
                0.06368156,
                0.06379731,
                0.07136511,
                0.07149489,
                0.07163991,
                0.07926588,
                0.07959762,
                0.08028712,
                0.087428,
                0.08755461,
                0.08780686,
                0.09508215,
                0.0955108,
                0.09573664,
                0.10264682,
                0.10346613,
                0.10410764,
                0.11041803,
                0.11116152,
                0.11142052,
                0.11764441,
                0.11897062,
                0.11937391,
                0.12732621,
                0.12748772,
                0.12775239,
                0.13439865,
                0.13516698,
                0.13527736,
                0.14320455,
                0.14322728,
                0.1432729,
                0.15046766,
                0.15117591,
                0.15288895,
                0.1567941,
                0.15912317,
                0.15974903,
                0.16605467,
                0.16685845,
                0.16706899,
                0.17442437,
                0.17501329,
                0.17697441,
                0.18169837,
                0.18295601,
                0.18569441,
                0.19089707,
                0.19111898,
                0.19146212,
                0.19805005,
                0.19883641,
                0.20174481,
                0.20451746,
                0.20677394,
                0.2084044,
                0.21386378,
                0.2147096,
                0.21783823,
                0.22264332,
                0.22311567,
                0.22673524,
                0.2281479,
                0.23004153,
                0.23057502,
            ]
        ),
    )

    probe.critical_edge(substrate=silicon, surface=air, n=25, delta=0.2)

    assert numpy.allclose(
        probe.calc_Q,
        numpy.array(
            [
                -0.00027897,
                0.0,
                0.00075572,
                0.00748785,
                0.0076777,
                0.00796095,
                0.00816768,
                0.00833784,
                0.008508,
                0.00867816,
                0.00884832,
                0.00901848,
                0.00918864,
                0.00935879,
                0.00952895,
                0.00969911,
                0.00986927,
                0.01003943,
                0.01020959,
                0.01037975,
                0.01054991,
                0.01072007,
                0.01089023,
                0.01106039,
                0.01123055,
                0.01140071,
                0.01157087,
                0.01174103,
                0.01191119,
                0.01208135,
                0.01225151,
                0.01487038,
                0.01592183,
                0.01624207,
                0.02317883,
                0.02388257,
                0.02441192,
                0.03130617,
                0.03184308,
                0.0321773,
                0.03904917,
                0.03980332,
                0.04107146,
                0.04717257,
                0.04753826,
                0.04776319,
                0.05572262,
                0.05606903,
                0.05635448,
                0.06275672,
                0.06368156,
                0.06379731,
                0.07136511,
                0.07149489,
                0.07163991,
                0.07926588,
                0.07959762,
                0.08028712,
                0.087428,
                0.08755461,
                0.08780686,
                0.09508215,
                0.0955108,
                0.09573664,
                0.10264682,
                0.10346613,
                0.10410764,
                0.11041803,
                0.11116152,
                0.11142052,
                0.11764441,
                0.11897062,
                0.11937391,
                0.12732621,
                0.12748772,
                0.12775239,
                0.13439865,
                0.13516698,
                0.13527736,
                0.14320455,
                0.14322728,
                0.1432729,
                0.15046766,
                0.15117591,
                0.15288895,
                0.1567941,
                0.15912317,
                0.15974903,
                0.16605467,
                0.16685845,
                0.16706899,
                0.17442437,
                0.17501329,
                0.17697441,
                0.18169837,
                0.18295601,
                0.18569441,
                0.19089707,
                0.19111898,
                0.19146212,
                0.19805005,
                0.19883641,
                0.20174481,
                0.20451746,
                0.20677394,
                0.2084044,
                0.21386378,
                0.2147096,
                0.21783823,
                0.22264332,
                0.22311567,
                0.22673524,
                0.2281479,
                0.23004153,
                0.23057502,
            ]
        ),
    )


if __name__ == "__main__":
    test()
