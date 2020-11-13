from .mock import MockObject, MockNetwork, MockUniformNetwork
from neet import Network, UniformNetwork
import unittest


class TestNetwork(unittest.TestCase):
    def test_is_network(self):
        self.assertTrue(isinstance(MockNetwork([3] * 5), Network))

        self.assertFalse(isinstance(MockObject(), Network))
        self.assertFalse(isinstance(5, Network))

    def test_invalid_names(self):
        with self.assertRaises(TypeError):
            MockNetwork([4, 3], names=5)
        with self.assertRaises(ValueError):
            MockNetwork([4, 3], names=['1'])
        with self.assertRaises(ValueError):
            MockNetwork([4, 3], names=['1', '2', '3'])

    def test_invalid_metadata(self):
        with self.assertRaises(TypeError):
            MockNetwork([4, 3], metadata=list())
        with self.assertRaises(TypeError):
            MockNetwork([4, 3], metadata='dict')

    def test_network_graph_invalid_labels(self):
        with self.assertRaises(ValueError):
            MockNetwork([4, 3]).network_graph(labels='anything')


class TestUniformNetwork(unittest.TestCase):
    def test_is_uniform_network(self):
        self.assertTrue(isinstance(MockUniformNetwork(5, 3), UniformNetwork))
        self.assertFalse(isinstance(MockNetwork([4] * 5), UniformNetwork))

    def test_base(self):
        self.assertTrue(MockUniformNetwork(5, 3).base, 3)

    def test_iter(self):
        net = MockUniformNetwork(2, 3)
        self.assertEqual(list(net), [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1],
                                     [2, 1], [0, 2], [1, 2], [2, 2]])

    def test_contains(self):
        net = MockUniformNetwork(2, 3)
        self.assertTrue([1, 1] in net)
        for state in net:
            self.assertTrue(state in net)
        self.assertFalse([-1, 0] in net)
        self.assertFalse([3, 0] in net)
        self.assertFalse([0, 3] in net)

        self.assertFalse(0 in net)
        self.assertFalse([2] in net)
        self.assertFalse([0, 0, 0] in net)

    def test_encode(self):
        net = MockUniformNetwork(2, 3)
        for i, state in enumerate(net):
            self.assertEqual(net.encode(state), i)

    def test_encode_decode(self):
        net = MockUniformNetwork(2, 3)
        for i, state in enumerate(net):
            self.assertEqual(state, net.decode(i))
            self.assertEqual(state, net.decode(net.encode(state)))
            self.assertEqual(i, net.encode(net.decode(i)))
