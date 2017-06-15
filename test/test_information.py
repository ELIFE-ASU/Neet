# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.automata import ECA
from neet.boolean.examples import s_pombe
from neet.information import active_information

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
