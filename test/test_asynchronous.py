# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.asynchronous import transitions
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
from .mock import MockObject

class TestAsync(unittest.TestCase):
    """
    Test the neet.asynchronous module
    """
    def test_canary(self):
        """
        A canary test
        """
        self.assertEqual(3, 1+2)

    def test_transitions_not_network(self):
        """
        Ensure that transitions fails when provided a non-network
        """
        with self.assertRaises(TypeError):
            list(transitions(5))

        with self.assertRaises(TypeError):
            list(transitions(MockObject()))

    def test_transitions_sum_to_one(self):
        """
        Ensure that the transition probabilities sum to one for each initial
        state
        """
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            for state in transitions(net):
                self.assertAlmostEqual(1.0, sum(state.values()))
