# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.interfaces import *
from neet.statespace import StateSpace
import neet.automata as ca
import neet.boolean as bnet
from neet.boolean.examples import s_pombe
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

        with self.assertRaises(ValueError):
            neighbors(eca, 1, direction='')

        self.assertTrue(neighbors(eca, 1, size=4), set([0, 1, 2]))

        with self.assertRaises(AttributeError):
            neighbors(eca, 1)

    def test_neighbors_WTNetwork(self):
        net = bnet.WTNetwork([[1,0],[1,1]])

        with self.assertRaises(ValueError):
            neighbors(net, 0, direction='')

        self.assertTrue(neighbors(net, 0), [set([0])])

    def test_neighbors_LogicNetwork(self):
        net = bnet.LogicNetwork([((0,), {'0'})])

        with self.assertRaises(ValueError):
            neighbors(net, 0, direction='')

        self.assertTrue(neighbors(net, 0), [set([0])])

    def test_neighbors_IsNetwork(self):
        net = self.IsNetwork()

    def test_to_networkx_graph_LogicNetwork(self):
        net = bnet.LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), ((0, 1), '10', [1, 1])),
                                    ((0, 1), {'11'})], ['A', 'B', 'C'])

        nx_net = to_networkx_graph(net,labels='names')
        self.assertEqual(set(nx_net),set(['A', 'B', 'C']))

    def test_to_networkx_graph_WTNetwork(self):

        nx_net = to_networkx_graph(s_pombe,labels='names')
        self.assertEqual(set(nx_net),set(s_pombe.names))

    def test_to_networkx_ECA_metadata(self):
        net = ca.ECA(30)
        net.boundary = (1,0)

        with self.assertRaises(AttributeError):
            to_networkx_graph(net)

        nx_net = to_networkx_graph(net,3)

        self.assertEqual(nx_net.graph['code'],30)
        self.assertEqual(nx_net.graph['size'],3)
        self.assertEqual(nx_net.graph['boundary'],(1,0))


    # def test_neighbors_IsNotNetwork(self):
    #     net = self.IsNotNetwork()
    #
    #     with self.assertRaises(AttributeError):
    #         neighbors(net)
