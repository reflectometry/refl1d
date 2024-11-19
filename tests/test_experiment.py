"""
Testing experiment serialization
"""

import os
import unittest

import numpy as np

from refl1d.names import QProbe, Slab, SLD, Parameter, Experiment, NeutronProbe, PolarizedNeutronProbe, Magnetism


class ExperimentJsonTest(unittest.TestCase):
    """Simple Experiment serialization test"""

    def test_save_json(self):
        """Check that we can produce a json file"""
        q_values = np.logspace(-2.1, -0.7, 100)
        # Fake data array
        zeros = np.zeros(len(q_values))
        ones = np.ones(len(q_values))
        # Q-resolution array
        q_resolution = q_values * 0.02 + 0.0001

        probe = QProbe(q_values, q_resolution, data=(zeros, ones))

        # Note: I don't use the usual shorthand to define the sample on purpose,
        #       so I see explicitly what I'm using.
        sample = (
            Slab(material=SLD(name="Si", rho=2.07, irho=0.0))
            | Slab(material=SLD(name="Cu", rho=6.5, irho=0.0), thickness=130, interface=15)
            | Slab(material=SLD(name="air", rho=0, irho=0.0))
        )

        sample["Cu"].thickness.range(90.0, 200.0)

        probe.intensity = Parameter(value=1.0, name="normalization")
        probe.background = Parameter(value=0.0, name="background")

        expt = Experiment(probe=probe, sample=sample)
        expt.save("output")

        self.assertTrue(os.path.isfile("output-expt.json"))


class ExperimentMagneticSimulateTest(unittest.TestCase):
    """Test the simulation functionality"""

    def setUp(self):
        self.q_values = np.logspace(-2.1, -0.6, 100)
        L = 4.75
        dL = 0.0475
        dT = 0.01

        xs = [NeutronProbe(T=self.q_values, dT=dT, L=L, dL=dL) for _ in range(4)]
        probe = PolarizedNeutronProbe(xs)

        # Note: I don't use the usual shorthand to define the sample on purpose,
        #       so I see explicitly what I'm using.
        # Note: 'MagneticSlab' is deprecated.  Using magnetism=... instead.
        sample = (
            Slab(material=SLD(name="Si", rho=2.07, irho=0.0))
            | Slab(material=SLD(name="Cu", rho=6.5, irho=0.0), thickness=130, interface=15)
            | Slab(
                SLD(name="Stuff", rho=4.0, irho=0.0),
                thickness=50.0,
                interface=1.0,
                magnetism=Magnetism(rhoM=0.2, thetaM=270),
            )
            | Slab(material=SLD(name="air", rho=0, irho=0.0))
        )

        sample["Cu"].thickness.range(90.0, 200.0)

        probe.intensity = Parameter(value=1.0, name="normalization")
        probe.background = Parameter(value=0.0, name="background")

        self.expt = Experiment(probe=probe, sample=sample)

        # Reference
        self.expt.simulate_data(noise=1e-11)
        self.r_i = self.expt.probe.pp.R[0]
        self.r_f = self.expt.probe.pp.R[-1]

    def test_none_noise_with_mag(self):
        """Provide None for noise with a magnetic sample"""

        self.expt.simulate_data(noise=None)
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.0)
        self.assertAlmostEqual(ratio_f, 0.0)

    def test_four_none_noise_with_mag(self):
        """Provide [None]*4 for noise with a magnetic sample"""

        self.expt.simulate_data(noise=[None] * 4)
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.0)
        self.assertAlmostEqual(ratio_f, 0.0)

    def test_noise_scalar_with_mag(self):
        """Provide a scalar of dR with a magnetic sample"""

        self.expt.simulate_data(noise=2.5)
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.025)
        self.assertAlmostEqual(ratio_f, 0.025)

    def test_four_noise_scalar_with_mag(self):
        """Provide four scalars of dR with a magnetic sample"""

        self.expt.simulate_data(noise=[2.5, 2.5, 2.5, 2.5])
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.025)
        self.assertAlmostEqual(ratio_f, 0.025)

    # unittest.skip("ambiguous interface not supported")
    def not_test_noise_array_with_mag(self):
        """Provide a dR array with a magnetic sample"""
        m = 10.0 / (self.q_values[-1] - self.q_values[0])
        b = 1 - m * self.q_values[0]
        _noise = m * self.q_values + b

        self.expt.simulate_data(noise=_noise)
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.01)
        self.assertAlmostEqual(ratio_f, 0.11)

    def test_four_noise_array_with_mag(self):
        """Provide four dR arrays with a magnetic sample"""
        m = 10.0 / (self.q_values[-1] - self.q_values[0])
        b = 1 - m * self.q_values[0]
        _noise = m * self.q_values + b

        self.expt.simulate_data(noise=[_noise, _noise, _noise, _noise])
        self.assertEqual(len(self.expt.probe.pp.dR), len(self.q_values))

        ratio_i = self.expt.probe.pp.dR[0] / self.r_i
        ratio_f = self.expt.probe.pp.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.01)
        self.assertAlmostEqual(ratio_f, 0.11)


class ExperimentNonMagSimulateTest(unittest.TestCase):
    """Test the simulation functionality"""

    def setUp(self):
        self.q_values = np.logspace(-2.1, -0.6, 100)
        L = 4.75
        dL = 0.0475
        dT = 0.01

        probe = NeutronProbe(T=self.q_values, dT=dT, L=L, dL=dL)

        # Note: I don't use the usual shorthand to define the sample on purpose,
        #       so I see explicitly what I'm using.
        sample = (
            Slab(material=SLD(name="Si", rho=2.07, irho=0.0))
            | Slab(material=SLD(name="Cu", rho=6.5, irho=0.0), thickness=130, interface=15)
            | Slab(material=SLD(name="air", rho=0, irho=0.0))
        )

        sample["Cu"].thickness.range(90.0, 200.0)

        probe.intensity = Parameter(value=1.0, name="normalization")
        probe.background = Parameter(value=0.0, name="background")

        self.expt = Experiment(probe=probe, sample=sample)

        # Reference
        self.expt.simulate_data(noise=1e-11)
        self.r_i = self.expt.probe.R[0]
        self.r_f = self.expt.probe.R[-1]

    def test_noise_scalar_non_mag(self):
        """Provide a scalar of dR with a non-magnetic sample"""

        self.expt.simulate_data(noise=2.5)
        self.assertEqual(len(self.expt.probe.dR), len(self.q_values))

        ratio_i = self.expt.probe.dR[0] / self.r_i
        ratio_f = self.expt.probe.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.025)
        self.assertAlmostEqual(ratio_f, 0.025)

    def test_noise_array_non_mag(self):
        """Provide an array of dR with a non-magnetic sample"""
        m = 10.0 / (self.q_values[-1] - self.q_values[0])
        b = 1 - m * self.q_values[0]
        _noise = m * self.q_values + b
        # print("noise input", _noise)

        self.expt.simulate_data(noise=_noise)
        self.assertEqual(len(self.expt.probe.dR), len(self.q_values))

        ratio_i = self.expt.probe.dR[0] / self.r_i
        ratio_f = self.expt.probe.dR[-1] / self.r_f

        self.assertAlmostEqual(ratio_i, 0.01)
        self.assertAlmostEqual(ratio_f, 0.11)


if __name__ == "__main__":
    unittest.main()
