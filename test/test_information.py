# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest
from neet.automata import ECA
from neet.boolean.examples import s_pombe
from neet.information import active_information, entropy_rate, transfer_entropy

class TestInformation(unittest.TestCase):
    """
    Test the information analysis module
    """
    def test_canary(self):
        """
        A canary test to ensure the test suite is working
        """
        self.assertEqual(3, 1+2)

    def test_active_info_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            active_information(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            active_information(5, k=3, timesteps=10, local=True)

    def test_active_info_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            active_information(ECA(30), k=3, timesteps=10, local=False)
        active_information(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_active_info_s_pombe(self):
        """
        ``active_information`` computes the correct values for ``s_pombe``
        """
        known_ai = [0.0, 0.408344, 0.629567, 0.629567, 0.379157, 0.400462,
                    0.670196, 0.670196, 0.391891]
        computed_ai = list(active_information(s_pombe, k=5, timesteps=20))
        self.assertEqual(9, len(computed_ai))
        for got, expected in zip(computed_ai, known_ai):
            self.assertAlmostEqual(expected, got, places=6)

    def test_entropy_rate_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            entropy_rate(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            entropy_rate(5, k=3, timesteps=10, local=True)

    def test_entropy_rate_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            entropy_rate(ECA(30), k=3, timesteps=10, local=False)
        entropy_rate(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_entropy_rate_s_pombe(self):
        """
        ``entropy_rate`` computes the correct values for ``s_pombe``
        """
        known_er = [0.0, 0.016912, 0.072803, 0.072803, 0.058420, 0.024794,
                    0.032173, 0.032173, 0.089669]
        computed_er = list(entropy_rate(s_pombe, k=5, timesteps=20))
        self.assertEqual(9, len(computed_er))
        for got, expected in zip(computed_er, known_er):
            self.assertAlmostEqual(expected, got, places=6)

    def test_transfer_entropy_not_network(self):
        """
        Raise a ``TypeError`` if the provided network is not actually a network
        """
        with self.assertRaises(TypeError):
            transfer_entropy(5, k=3, timesteps=10, local=False)
        with self.assertRaises(TypeError):
            transfer_entropy(5, k=3, timesteps=10, local=True)

    def test_transfer_entropy_not_fixed_size(self):
        """
        Raise a ``ValueError`` if the provided network is not fixed sized, and
        the ``size`` argument is ``None``
        """
        with self.assertRaises(ValueError):
            transfer_entropy(ECA(30), k=3, timesteps=10, local=False)
        transfer_entropy(ECA(30), k=3, timesteps=10, size=5, local=False)

    def test_transfer_entropy_s_pombe(self):
        """
        ``transfer_entropy`` computes the correct values for ``s_pombe``
        """
        known_te = np.asarray(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.016912, 0., 0., 0., 0.],
             [0., 0.051370, 0., 0.012225, 0.019947, 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.051370, 0.012225, 0., 0.019947, 0.051370, 0.006039, 0.006039, 0.072803],
             [0., 0.058420, 0.047602, 0.047602, 0., 0.058420, 0.047602, 0.047602, 0.],
             [0., 0., 0.024794, 0.024794, 0., 0., 0.024794, 0.024794, 0.],
             [0., 0.016690, 0.004526, 0.004526, 0.011916, 0.016690, 0., 0.002983, 0.032173],
             [0., 0.016690, 0.004526, 0.004526, 0.011916, 0.016690, 0.002983, 0., 0.032173],
             [0., 0.060304, 0.048289, 0.048289, 0.089669, 0.060304, 0.048927, 0.048927, 0.]])
        computed_te = transfer_entropy(s_pombe, k=5, timesteps=20)
        self.assertEqual(known_te.shape, computed_te.shape)
        for got, expected in zip(computed_te.flatten(), known_te.flatten()):
            self.assertAlmostEqual(expected, got, places=6)
