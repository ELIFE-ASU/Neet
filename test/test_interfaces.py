import unittest
from neet.interfaces import (Network, is_fixed_sized, is_boolean_network, neighbors,
                             to_networkx_graph)
from neet.statespace import StateSpace
import neet.automata as ca
import neet.boolean as bnet
from neet.boolean.examples import s_pombe


class TestCore(unittest.TestCase):
    class IsNetwork(Network):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1)

        def neighbors(self):
            pass

        @property
        def size(self):
            return 0

    Network.register(IsNetwork)

    class FixedSizeNetwork(IsNetwork):
        pass

    Network.register(IsNetwork)

    class IsNotNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1)

        def neighbors(self):
            pass

    class NotFixedSizedNetwork(IsNotNetwork):
        def size(self):
            return 5

    class BaseThreeNetwork(Network):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1, base=3)

        def neighbors(self):
            pass

        @property
        def size(self):
            return 0

    Network.register(BaseThreeNetwork)

    class MultipleBaseNetwork(Network):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace([1, 2, 3])

        def neighbors(self):
            pass

        @property
        def size(self):
            return 0

    Network.register(MultipleBaseNetwork)

    def test_is_network(self):
        net = self.IsNetwork()
        self.assertTrue(isinstance(net, Network))

        not_net = self.IsNotNetwork()
        self.assertFalse(isinstance(not_net, Network))

        self.assertFalse(isinstance(5, Network))

    def test_is_fixed_sized(self):
        net = self.IsNetwork()
        self.assertTrue(is_fixed_sized(net))

        not_net = self.IsNotNetwork()
        self.assertFalse(is_fixed_sized(not_net))

        net = self.FixedSizeNetwork()
        self.assertTrue(is_fixed_sized(net))

        not_net = self.NotFixedSizedNetwork()
        self.assertFalse(is_fixed_sized(not_net))

    def test_is_boolean_network(self):
        net = self.IsNetwork()
        self.assertTrue(is_boolean_network(net))

        not_bool_net = self.BaseThreeNetwork()
        self.assertFalse(is_boolean_network(not_bool_net))

        not_bool_net = self.MultipleBaseNetwork()
        self.assertFalse(is_boolean_network(not_bool_net))

    def test_neighbors_ECA(self):
        eca = ca.ECA(30, 4)

        with self.assertRaises(ValueError):
            neighbors(eca, 1, direction='')

        self.assertTrue(neighbors(eca, 1), set([0, 1, 2]))

    def test_neighbors_WTNetwork(self):
        net = bnet.WTNetwork([[1, 0], [1, 1]])

        with self.assertRaises(ValueError):
            neighbors(net, 0, direction='')

        self.assertTrue(neighbors(net, 0), [set([0])])

    def test_neighbors_LogicNetwork(self):
        net = bnet.LogicNetwork([((0,), {'0'})])

        with self.assertRaises(ValueError):
            neighbors(net, 0, direction='')

        self.assertTrue(neighbors(net, 0), [set([0])])

    def test_to_networkx_graph_LogicNetwork(self):
        net = bnet.LogicNetwork([((1, 2), {'01', '10'}),
                                 ((0, 2), ((0, 1), '10', [1, 1])),
                                 ((0, 1), {'11'})], ['A', 'B', 'C'])

        nx_net = to_networkx_graph(net, labels='names')
        self.assertEqual(set(nx_net), set(['A', 'B', 'C']))

    def test_to_networkx_graph_WTNetwork(self):

        nx_net = to_networkx_graph(s_pombe, labels='names')
        self.assertEqual(set(nx_net), set(s_pombe.names))

    def test_to_networkx_ECA_metadata(self):
        net = ca.ECA(30, 3)
        net.boundary = (1, 0)

        nx_net = to_networkx_graph(net)

        self.assertEqual(nx_net.graph['code'], 30)
        self.assertEqual(nx_net.graph['size'], 3)
        self.assertEqual(nx_net.graph['boundary'], (1, 0))
