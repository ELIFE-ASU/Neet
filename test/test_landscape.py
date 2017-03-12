# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.landscape import *
import numpy as np

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return StateSpace(1)

    class IsNotNetwork(object):
        pass

    def test_trajectory_not_network(self):
        with self.assertRaises(TypeError):
            list(trajectory(5, [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(self.IsNotNetwork(), [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(self.IsNetwork, [1,2,3]))

    def test_trajectory_too_short(self):
        with self.assertRaises(ValueError):
            list(trajectory(self.IsNetwork(), [1,2,3], n=0))

        with self.assertRaises(ValueError):
            list(trajectory(self.IsNetwork(), [1,2,3], n=-1))

    def test_trajectory_eca_not_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, []))

        xs = [0,1,0]
        got = list(trajectory(rule30, xs))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([[0,1,0],[1,1,1]], got)

        got = list(trajectory(rule30, xs, n=2))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([[0,1,0],[1,1,1],[0,0,0]], got)

    def test_trajectory_eca_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, [], encode=True))

        xs = [0,1,0]
        got = list(trajectory(rule30, xs, encode=True))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([2,7], got)

        got = list(trajectory(rule30, xs, n=2, encode=True))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([2,7,0], got)


    def test_transitions_not_network(self):
        with self.assertRaises(TypeError):
            list(transitions(self.IsNotNetwork(), StateSpace(5)))

    def test_transitions_not_statespace(self):
        with self.assertRaises(TypeError):
            list(transitions(self.IsNetwork(), 5))

    def test_transitions_eca_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)

        got = list(transitions(rule30, n=1))
        self.assertEqual([0,0], got)

        got = list(transitions(rule30, n=2))
        self.assertEqual([0,1,2,0], got)

        got = list(transitions(rule30, n=3))
        self.assertEqual([0,7,7,1,7,4,2,0], got)

    def test_transitions_eca_not_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)

        got = list(transitions(rule30, n=1, encode=False))
        self.assertEqual([[0],[0]], got)

        got = list(transitions(rule30, n=2, encode=False))
        self.assertEqual([[0,0],[1,0],[0,1],[0,0]], got)

        got = list(transitions(rule30, n=3, encode=False))
        self.assertEqual([[0,0,0],[1,1,1],[1,1,1],[1,0,0]
                         ,[1,1,1],[0,0,1],[0,1,0],[0,0,0]], got)
