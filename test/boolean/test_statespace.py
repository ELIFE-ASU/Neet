import unittest
from neet.python import long
from neet.boolean.statespace import BooleanSpace
import numpy as np


class TestBooleanSpace(unittest.TestCase):
    def assertArrayEqual(self, a, b):
        self.assertTrue(np.array_equal(a, b))

    def test_invalid_ndim_type(self):
        with self.assertRaises(TypeError):
            BooleanSpace('a')

        with self.assertRaises(TypeError):
            BooleanSpace([2])

        with self.assertRaises(TypeError):
            BooleanSpace(3.0)

    def test_invalid_shape_value(self):
        with self.assertRaises(ValueError):
            BooleanSpace(0)

        with self.assertRaises(ValueError):
            BooleanSpace(-1)

    def test_shape(self):
        space = BooleanSpace(5)
        self.assertEqual(2, space.base)
        self.assertEqual(32, space.volume)
        self.assertEqual(5, space.ndim)
        self.assertEqual([2, 2, 2, 2, 2], space.shape)

    def test_states(self):
        space = BooleanSpace(1)
        self.assertEqual([[0], [1]], list(space))

        space = BooleanSpace(2)
        self.assertEqual([[0, 0], [1, 0], [0, 1], [1, 1]], list(space))

        space = BooleanSpace(3)
        self.assertEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                         list(space))

    def test_states_count(self):
        space = BooleanSpace(5)
        count = 0
        for state in space:
            count += 1
        self.assertEqual(32, count)

    def test_encoding_error(self):
        space = BooleanSpace(3)
        with self.assertRaises(ValueError):
            space.encode([1, 1])

        space = BooleanSpace(1)
        with self.assertRaises(ValueError):
            space.encode([2])

        space = BooleanSpace(2)
        with self.assertRaises(ValueError):
            space.encode([1, 3])

        with self.assertRaises(ValueError):
            space.encode([1, -1])

    def test_encoding(self):
        for ndim in range(1, 5):
            space = BooleanSpace(ndim)
            counter = 0
            for state in space:
                encoding = space.encode(state)
                self.assertEqual(counter, encoding)
                counter += 1

    def test_decoding(self):
        for ndim in range(1, 5):
            space = BooleanSpace(ndim)
            states = list(space)
            decoded = list(map(space.decode, range(space.volume)))
            self.assertEqual(states, decoded)

    def test_encode_decode(self):
        for ndim in range(1, 5):
            space = BooleanSpace(ndim)
            for state in space:
                encoded = space.encode(state)
                decoded = space.decode(encoded)
                self.assertEqual(state, decoded)

    def test_decode_encode(self):
        for ndim in range(1, 5):
            space = BooleanSpace(ndim)
            for i in range(2**ndim):
                decoded = space.decode(i)
                encoded = space.encode(decoded)
                self.assertEqual(i, encoded)

    def test_check_states(self):
        state_space = BooleanSpace(3)
        self.assertTrue([0, 1, 1] in state_space)
        self.assertFalse([0, 0] in state_space)
        self.assertFalse([1, 2, 0] in state_space)

        self.assertFalse([0, 1, 1] not in state_space)
        self.assertTrue([0, 0] not in state_space)
        self.assertTrue([1, 2, 0] not in state_space)

        self.assertFalse(1 in state_space)
        self.assertFalse("string" in state_space)

    def test_long_encoding(self):
        state_space = BooleanSpace(10)
        code = state_space.encode(np.ones(10, dtype=int))
        self.assertIsInstance(code, long)

        state_space = BooleanSpace(68)
        code = state_space.encode(np.ones(68, dtype=int))
        self.assertIsInstance(code, long)

        state_space = BooleanSpace(100)
        code = state_space.encode(np.ones(100, dtype=int))
        self.assertIsInstance(code, long)

    def test_subspace_invalid_indices(self):
        space = BooleanSpace(3)

        with self.assertRaises(TypeError):
            list(space.subspace(3))

        with self.assertRaises(Exception):
            list(space.subspace('abc'))

        with self.assertRaises(IndexError):
            list(space.subspace([-1]))

        with self.assertRaises(IndexError):
            list(space.subspace([0, -1]))

        with self.assertRaises(IndexError):
            list(space.subspace([10]))

        with self.assertRaises(IndexError):
            list(space.subspace([0, 10]))

    def test_subspace_invalid_state(self):
        space = BooleanSpace(3)
        with self.assertRaises(ValueError):
            list(space.subspace([0, 1], [0, 0]))

        with self.assertRaises(ValueError):
            list(space.subspace([0, 1], [0, 0, 0, 0]))

    def test_subspace_no_indices(self):
        space = BooleanSpace(3)
        self.assertEqual(list(space.subspace([])), [[0, 0, 0]])
        self.assertEqual(list(space.subspace([], [1, 1, 0])), [[1, 1, 0]])

    def test_subspace_all_indices(self):
        space = BooleanSpace(3)
        self.assertEqual(list(space.subspace([0, 1, 2])), list(space))
        self.assertEqual(list(space.subspace([1, 2, 2, 0, 1])), list(space))
        self.assertEqual(list(space.subspace([0, 1, 2], [0, 1, 0])), list(space))
        self.assertEqual(list(space.subspace([1, 2, 2, 0, 1], [0, 1, 0])), list(space))

    def test_subspace(self):
        space = BooleanSpace(3)
        self.assertEqual(list(space.subspace([1])),
                         [[0, 0, 0], [0, 1, 0]])
        self.assertEqual(list(space.subspace([1, 1])),
                         [[0, 0, 0], [0, 1, 0]])

        self.assertEqual(list(space.subspace([1], [1, 1, 0])),
                         [[1, 1, 0], [1, 0, 0]])
        self.assertEqual(list(space.subspace([1, 1], [1, 1, 0])),
                         [[1, 1, 0], [1, 0, 0]])

        self.assertEqual(list(space.subspace([0, 2])),
                         [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
        self.assertEqual(list(space.subspace([0, 2, 2])),
                         [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])

        self.assertEqual(list(space.subspace([0, 2], [1, 1, 0])),
                         [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]])
        self.assertEqual(list(space.subspace([0, 2, 0], [1, 1, 0])),
                         [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]])

    def test_subspace_numpy(self):
        space = BooleanSpace(3)
        self.assertArrayEqual(list(space.subspace([1], np.asarray([1, 1, 0]))),
                              [[1, 1, 0], [1, 0, 0]])
        self.assertArrayEqual(list(space.subspace([1, 1], np.asarray([1, 1, 0]))),
                              [[1, 1, 0], [1, 0, 0]])

        self.assertArrayEqual(list(space.subspace([0, 2], np.asarray([1, 1, 0]))),
                              [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]])
        self.assertArrayEqual(list(space.subspace([0, 2, 0], np.asarray([1, 1, 0]))),
                              [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]])

    def test_hamming_neighbors_input(self):
        space = BooleanSpace(3)

        with self.assertRaises(ValueError):
            list(space.hamming_neighbors([0, 0]))

        with self.assertRaises(ValueError):
            list(space.hamming_neighbors([0, 0, 0, 0]))

        with self.assertRaises(ValueError):
            list(space.hamming_neighbors([0, 1, 2]))

        with self.assertRaises(ValueError):
            list(space.hamming_neighbors([[0, 0, 1], [1, 0, 0]]))

    def test_hamming_neighbors_example(self):
        space = BooleanSpace(4)

        state = [0, 1, 1, 0]
        neighbors = [[1, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 1]]
        self.assertTrue(neighbors, list(space.hamming_neighbors(state)))

        state = np.asarray(state)
        self.assertTrue(neighbors, list(space.hamming_neighbors(state)))

    def test_distance_raises(self):
        space = BooleanSpace(3)

        with self.assertRaises(ValueError):
            space.distance([0, 0], [0, 0, 0])

        with self.assertRaises(ValueError):
            space.distance([0, 0, 0, 0], [0, 0, 0])

        with self.assertRaises(ValueError):
            space.distance([0, 0, 0], [0, 0])

        with self.assertRaises(ValueError):
            space.distance([0, 0, 0], [0, 0, 0, 0])

        with self.assertRaises(ValueError):
            space.distance([0, 1, 2], [0, 1, 1])

        with self.assertRaises(ValueError):
            space.distance([0, 1, 1], [0, 1, 2])

    def test_distance(self):
        space = BooleanSpace(3)

        self.assertEqual(0, space.distance([0, 1, 1], [0, 1, 1]))
        self.assertEqual(1, space.distance([0, 0, 0], [1, 0, 0]))
        self.assertEqual(1, space.distance([0, 0, 0], [0, 1, 0]))
        self.assertEqual(1, space.distance([0, 0, 0], [0, 0, 1]))
        self.assertEqual(2, space.distance([0, 1, 0], [1, 0, 0]))
        self.assertEqual(2, space.distance([1, 0, 0], [0, 1, 0]))
