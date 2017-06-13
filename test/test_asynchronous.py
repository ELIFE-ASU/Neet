# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.asynchronous import *
from neet.boolean.examples import s_pombe

class TestAsync(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass
        @property
        def size(self):
            pass
        def state_space(self):
            return StateSpace(1)

    class IsNotNetwork(object):
        pass

    def test_canary(self):
        self.assertEqual(3, 1+2)

    def test_transitions_not_network(self):
        with self.assertRaises(TypeError):
            list(transitions(5))

        with self.assertRaises(TypeError):
            list(transitions(self.IsNotNetwork()))

        with self.assertRaises(TypeError):
            list(transitions(self.IsNetwork))

    def test_transitions_spombe(self):
        for state in transitions(s_pombe):
            self.assertAlmostEqual(1.0, sum(state.values()))