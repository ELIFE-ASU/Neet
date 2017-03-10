# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet
import numpy as np

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass

    class IsNotNetwork(object):
        pass

    def test_is_network(self):
        net = self.IsNetwork()
        self.assertTrue(neet.is_network(net))
        self.assertFalse(neet.is_network(self.IsNetwork))

        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_network(not_net))
        self.assertFalse(neet.is_network(self.IsNotNetwork))

        self.assertFalse(neet.is_network(5))
        self.assertFalse(neet.is_network(int))

    def test_is_network_type(self):
        net = self.IsNetwork()
        self.assertFalse(neet.is_network_type(net))
        self.assertTrue(neet.is_network_type(self.IsNetwork))

        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_network_type(not_net))
        self.assertFalse(neet.is_network_type(self.IsNotNetwork))

        self.assertFalse(neet.is_network_type(5))
        self.assertFalse(neet.is_network_type(int))

    def test_trajectory_not_network(self):
        with self.assertRaises(TypeError):
            neet.trajectory(5, [1,2,3])

        with self.assertRaises(TypeError):
            neet.trajectory(self.IsNotNetwork(), [1,2,3])

        with self.assertRaises(TypeError):
            neet.trajectory(self.IsNetwork, [1,2,3])

    def test_trajectory_too_short(self):
        with self.assertRaises(ValueError):
            neet.trajectory(self.IsNetwork(), [1,2,3], n=0)

        with self.assertRaises(ValueError):
            neet.trajectory(self.IsNetwork(), [1,2,3], n=-1)

    def test_trajectory_eca(self):
        from neet.automata import ECA
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            neet.trajectory(rule30, [])

        xs = [0,1,0]
        got = neet.trajectory(rule30, xs)
        self.assertEqual([0,1,0], xs)
        self.assertTrue(np.array_equal([[0,1,0],[1,1,1]], got))

        got = neet.trajectory(rule30, xs, n=2)
        self.assertEqual([0,1,0], xs)
        self.assertTrue(np.array_equal([[0,1,0],[1,1,1],[0,0,0]], got))