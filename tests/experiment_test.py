"""
    Testing experiment serialization
"""
from __future__ import absolute_import, division, print_function
import unittest
import os
import numpy as np
from refl1d.names import QProbe, Slab, SLD, Parameter, Experiment

class ExperimentJsonTest(unittest.TestCase):
    def test_save_json(self):
        q_values = np.logspace(-2.1, -.7, 100)
        # Fake data array
        zeros = np.zeros(len(q_values))
        ones = np.ones(len(q_values))
        # Q-resolution array
        dq = q_values * 0.02 + 0.0001

        probe = QProbe(q_values, dq, data=(zeros, ones))

        # Note: I don't use the usual shorthand to define the sample on purpose, so see
        #        explicitly what I'm using.
        sample = Slab(material=SLD(name='Si', rho=2.07, irho=0.0)) \
            | Slab(material=SLD(name='Cu', rho=6.5, irho=0.0), thickness=130, interface=15) \
            | Slab(material=SLD(name='air', rho=0, irho=0.0))

        sample['Cu'].thickness.range(90.0, 200.0)

        probe.intensity=Parameter(value=1.0, name='normalization')
        probe.background=Parameter(value=0.0,name='background')

        expt = Experiment(probe=probe, sample=sample)
        expt.save('output')

        # TODO: expand this
        self.assertTrue(os.path.isfile('output-expt.json'))

if __name__ == '__main__':
    unittest.main()

