# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.interfaces import *
from neet.statespace import StateSpace
import neet.automata as ca
import neet.boolean as bnet
import numpy as np


class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1)

        def neighbors(self):
            pass

    class FixedSizeNetwork(IsNetwork):
        def size(self):
            return 5

    class IsNotNetwork(object):
        pass

    class NotFixedSizedNetwork(IsNotNetwork):
        def size(self):
            return 5

    class BaseThreeNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1, base=3)

        def neighbors(self):
            pass

    class MultipleBaseNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace([1, 2, 3])

        def neighbors(self):
            pass

    def test_is_network(self):
        net = self.IsNetwork()
        self.assertTrue(is_network(net))
        self.assertTrue(is_network(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(is_network(not_net))
        self.assertFalse(is_network(type(not_net)))

        self.assertFalse(is_network(5))
        self.assertFalse(is_network(int))

    def test_is_fixed_sized(self):
        net = self.IsNetwork()
        self.assertFalse(is_fixed_sized(net))
        self.assertFalse(is_fixed_sized(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(is_fixed_sized(not_net))
        self.assertFalse(is_fixed_sized(type(not_net)))

        net = self.FixedSizeNetwork()
        self.assertTrue(is_fixed_sized(net))
        self.assertTrue(is_fixed_sized(type(net)))

        not_net = self.NotFixedSizedNetwork()
        self.assertFalse(is_fixed_sized(not_net))
        self.assertFalse(is_fixed_sized(type(not_net)))

    def test_is_boolean_network(self):
        net = self.IsNetwork()
        self.assertTrue(is_boolean_network(net))

        not_bool_net = self.BaseThreeNetwork()
        self.assertFalse(is_boolean_network(not_bool_net))

        not_bool_net = self.MultipleBaseNetwork()
        self.assertFalse(is_boolean_network(not_bool_net))

    def test_neighbors_ECA(self):
        eca = ca.ECA(30)

        self.assertEqual(neighbors(eca,size=4),[set([0, 1, 3]),
                                               set([0, 1, 2]), 
                                               set([1, 2, 3]), 
                                               set([0, 2, 3])])

        with self.assertRaises(AttributeError):
            neighbors(eca)

    def test_neighbors_WTNetwork(self):
        net = bnet.WTNetwork([[1,0],[1,1]])

        self.assertEqual(neighbors(net),[set([0,1]),set([0,1])])
        # test kwargs
        self.assertEqual(neighbors(net,direction='in'),
                        [set([0]),set([0,1])])
        self.assertEqual(neighbors(net,direction='out'),
                        [set([0,1]),set([1])])
        self.assertEqual(neighbors(net,index=0),set([0,1]))

    def test_neighbors_LogicNetwork(self):
        net = bnet.LogicNetwork([((0,1), {'00', '11'}), ((1,), {'1'})])

        self.assertEqual(neighbors(net),[set([0,1]),set([0,1])])
        # test kwargs
        self.assertEqual(neighbors(net,direction='in'),
                         [set([0,1]),set([1])])
        self.assertEqual(neighbors(net,direction='out'),
                         [set([0]),set([0,1])])
        self.assertEqual(neighbors(net,index=0),set([0,1]))

    def test_neighbors_IsNetwork(self):
        net = self.IsNetwork()

        # with self.assertRaises(AttributeError):
        #     neighbors(net)

    def test_neighbors_IsNotNetwork(self):
        net = self.IsNotNetwork()

        with self.assertRaises(AttributeError):
            neighbors(net)



