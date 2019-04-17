import unittest
from neet.interfaces import (Network, is_boolean_network, neighbors,
                             to_networkx_graph)
import neet.automata as ca
import neet.boolean as bnet
from neet.statespace import StateSpace
from neet.boolean.examples import s_pombe
from .mock import MockObject, MockNetwork


class MockBaseThreeNetwork(MockNetwork):
    def __init__(self, size):
        super(MockBaseThreeNetwork, self).__init__(size)

    def state_space(self):
        return StateSpace(self.size, base=3)


MockNetwork.register(MockBaseThreeNetwork)


class MockMultipleBaseNetwork(MockNetwork):
    def __init__(self, size):
        super(MockMultipleBaseNetwork, self).__init__(size)

    def state_space(self):
        return StateSpace([1, 2, 3])


MockNetwork.register(MockMultipleBaseNetwork)


class TestCore(unittest.TestCase):
    def test_is_network(self):
        net = MockNetwork(5)
        self.assertTrue(isinstance(net, Network))

        not_net = MockObject()
        self.assertFalse(isinstance(not_net, Network))

        self.assertFalse(isinstance(5, Network))

    def test_is_boolean_network(self):
        net = MockNetwork(5)
        self.assertTrue(is_boolean_network(net))

        not_bool_net = MockBaseThreeNetwork(5)
        self.assertFalse(is_boolean_network(not_bool_net))

        not_bool_net = MockMultipleBaseNetwork(5)
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
