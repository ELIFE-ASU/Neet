from ..mock import MockNetwork, MockUniformNetwork, MockBooleanNetwork
from neet.boolean import BooleanNetwork
import unittest


class TestBooleanNetwork(unittest.TestCase):
    def test_is_boolean_network(self):
        self.assertFalse(isinstance(MockUniformNetwork(5, 2), BooleanNetwork))
        self.assertFalse(isinstance(MockNetwork([4] * 5), BooleanNetwork))
        self.assertTrue(isinstance(MockBooleanNetwork(5), BooleanNetwork))

    def test_base(self):
        self.assertTrue(MockBooleanNetwork(5).base, 2)

    def test_iter(self):
        net = MockBooleanNetwork(2)
        self.assertEqual(list(net), [[0, 0], [1, 0], [0, 1], [1, 1]])

    def test_contains(self):
        net = MockBooleanNetwork(2)
        self.assertTrue([1, 1] in net)
        for state in net:
            self.assertTrue(state in net)
        self.assertFalse([-1, 0] in net)
        self.assertFalse([2, 0] in net)
        self.assertFalse([0, 2] in net)

        self.assertFalse(0 in net)
        self.assertFalse([2] in net)
        self.assertFalse([0, 0, 0] in net)

    def test_encode(self):
        net = MockBooleanNetwork(2)
        for i, state in enumerate(net):
            self.assertEqual(net.encode(state), i)

    def test_encode_decode(self):
        net = MockBooleanNetwork(2)
        for i, state in enumerate(net):
            self.assertEqual(state, net.decode(i))
            self.assertEqual(state, net.decode(net.encode(state)))
            self.assertEqual(i, net.encode(net.decode(i)))

    def test_subspace(self):
        net = MockBooleanNetwork(3)
        self.assertEqual(list(net.subspace([0])), [[0, 0, 0], [1, 0, 0]])
        self.assertEqual(list(net.subspace([1])), [[0, 0, 0], [0, 1, 0]])
        self.assertEqual(list(net.subspace([0, 1])), [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.assertEqual(list(net.subspace([0, 1, 2])), list(net))

        self.assertEqual(list(net.subspace([0], state=[0, 1, 1])), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(list(net.subspace([1], state=[0, 1, 1])), [[0, 1, 1], [0, 0, 1]])
        self.assertEqual(list(net.subspace([0, 1], state=[0, 1, 1])),
                         [[0, 1, 1], [1, 1, 1], [0, 0, 1], [1, 0, 1]])
        self.assertEqual(set(map(tuple, net.subspace([0, 1, 2], state=[0, 1, 1]))),
                         set(map(tuple, net)))

    def test_subspace_raises(self):
        net = MockBooleanNetwork(3)
        with self.assertRaises(ValueError):
            list(net.subspace([0], state=[0, 0, 0, 0]))
        with self.assertRaises(ValueError):
            list(net.subspace([0], state=[0, 0]))
        with self.assertRaises(ValueError):
            list(net.subspace([0], state=[2, 0, 0]))
        with self.assertRaises(ValueError):
            list(net.subspace([0], state=0))

        with self.assertRaises(IndexError):
            list(net.subspace([-1]))
        with self.assertRaises(IndexError):
            list(net.subspace([3]))

    def test_hamming_neighbors(self):
        net = MockBooleanNetwork(3)
        self.assertEqual(net.hamming_neighbors([0, 0, 0]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(net.hamming_neighbors([0, 1, 0]), [[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        for state in net:
            for neighbor in net.hamming_neighbors(state):
                self.assertTrue(neighbor in net)
                self.assertNotEqual(neighbor, state)

    def test_hamming_neighbors_raises(self):
        net = MockBooleanNetwork(3)
        with self.assertRaises(ValueError):
            net.hamming_neighbors(0)
        with self.assertRaises(ValueError):
            net.hamming_neighbors([0, 0])
        with self.assertRaises(ValueError):
            net.hamming_neighbors([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            net.hamming_neighbors([2, 0, 0])
        with self.assertRaises(ValueError):
            net.hamming_neighbors([-1, 0, 0])

    def test_distance(self):
        net = MockBooleanNetwork(3)
        for state in net:
            self.assertEqual(net.distance(state, state), 0)
        for state in net:
            self.assertEqual(net.distance([0, 0, 0], state), sum(state))
            self.assertEqual(net.distance(state, [0, 0, 0]), sum(state))
        for state in net:
            self.assertEqual(net.distance([1, 1, 1], state), len(state) - sum(state))
            self.assertEqual(net.distance(state, [1, 1, 1]), len(state) - sum(state))
        for x in net:
            for y in net:
                self.assertEqual(net.distance(x, y), net.distance(y, x))

    def test_distance_raises(self):
        net = MockBooleanNetwork(3)
        with self.assertRaises(ValueError):
            net.distance(0, [0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([2, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([-1, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0], 0)
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0], [0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0], [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0], [2, 0, 0])
        with self.assertRaises(ValueError):
            net.distance([0, 0, 0], [-1, 0, 0])
