# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.asynchronous import transitions
from neet.automata import ECA
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
            for _, probabilities in transitions(net):
                self.assertGreaterEqual(net.size, len(probabilities))
                self.assertAlmostEqual(1.0, sum(probabilities))

        for net in [ECA(30), ECA(110), ECA(42)]:
            for size in [5, 8, 10]:
                for _, probabilities in transitions(net, size):
                    self.assertGreaterEqual(size, len(probabilities))
                    self.assertAlmostEqual(1.0, sum(probabilities))

    def test_transitions_require_update(self):
        """
        Ensure that the transition probabilities sum to one for each initial
        state
        """
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            for _, probabilities in transitions(net, require_update=True):
                self.assertGreaterEqual(net.size, len(probabilities))
                if len(probabilities) != 0:
                    self.assertAlmostEqual(1.0, sum(probabilities))

        for net in [ECA(30), ECA(110), ECA(42)]:
            for size in [5, 8, 10]:
                for _, probabilities in transitions(net, size, require_update=True):
                    self.assertGreaterEqual(size, len(probabilities))
                    if len(probabilities) != 0:
                        self.assertAlmostEqual(1.0, sum(probabilities))

    def test_transitions_encoded(self):
        """
        Ensure that the transitions function's encoded keyword works
        """
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            for states, _ in transitions(net, encoded=True):
                for state in states:
                    self.assertIsInstance(state, (int, long))
            for states, _ in transitions(net, encoded=False):
                for state in states:
                    self.assertIsInstance(state, list)

        for net in [ECA(30), ECA(110), ECA(42)]:
            for size in [5, 8, 10]:
                for states, _ in transitions(net, size, encoded=True):
                    for state in states:
                        self.assertIsInstance(state, (int, long))
                for states, _ in transitions(net, size, encoded=False):
                    for state in states:
                        self.assertIsInstance(state, list)
