from .mock import MockObject, MockNetwork, MockBooleanNetwork
from neet import Network
from neet.boolean import BooleanNetwork, ECA, LogicNetwork, WTNetwork
from neet.boolean.examples import s_pombe
import unittest


class TestNetwork(unittest.TestCase):
    def test_is_network(self):
        net = MockNetwork([3, 3, 3, 3, 3])
        self.assertTrue(isinstance(net, Network))

        not_net = MockObject()
        self.assertFalse(isinstance(not_net, Network))

        self.assertFalse(isinstance(5, Network))

    def test_is_boolean_network(self):
        self.assertTrue(isinstance(MockBooleanNetwork(5), BooleanNetwork))

        self.assertFalse(isinstance(MockNetwork([4, 3, 2, 4, 5]), BooleanNetwork))

    def test_neighbors_ECA(self):
        eca = ECA(30, 4)

        with self.assertRaises(ValueError):
            eca.neighbors(1, direction='')

        self.assertTrue(eca.neighbors(1), set([0, 1, 2]))

    def test_neighbors_WTNetwork(self):
        net = WTNetwork([[1, 0], [1, 1]])

        with self.assertRaises(ValueError):
            net.neighbors(0, direction='')

        self.assertTrue(net.neighbors(0), [set([0])])

    def test_neighbors_LogicNetwork(self):
        net = LogicNetwork([((0,), {'0'})])

        with self.assertRaises(ValueError):
            net.neighbors(0, direction='')

        self.assertTrue(net.neighbors(0), [set([0])])

    def test_network_graph_LogicNetwork(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                                 ((0, 2), ((0, 1), '10', [1, 1])),
                                 ((0, 1), {'11'})], ['A', 'B', 'C'])

        nx_net = net.network_graph(labels='names', title='Logic Network')
        self.assertEqual(set(nx_net), set(['A', 'B', 'C']))
        self.assertEqual(nx_net.graph['title'], 'Logic Network')

    def test_network_graph_WTNetwork(self):
        nx_net = s_pombe.network_graph(labels='names', title='S. pombe')
        self.assertEqual(set(nx_net), set(s_pombe.names))
        self.assertEqual(nx_net.graph['name'], 's_pombe')
        self.assertEqual(nx_net.graph['title'], 'S. pombe')

    def test_to_networkx_ECA_metadata(self):
        net = ECA(30, 3)
        net.boundary = (1, 0)

        nx_net = net.network_graph()

        self.assertEqual(nx_net.graph['code'], 30)
        self.assertEqual(nx_net.graph['boundary'], (1, 0))
