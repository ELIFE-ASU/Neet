import unittest
from neet.python import long
from neet.statespace import StateSpace
import numpy as np


class TestStateSpace(unittest.TestCase):
    def test_invalid_spec_type(self):
        with self.assertRaises(TypeError):
            StateSpace("a")

        with self.assertRaises(TypeError):
            StateSpace("abc")

    def test_invalid_base_type(self):
        with self.assertRaises(TypeError):
            StateSpace(5, base='a')

        with self.assertRaises(TypeError):
            StateSpace(3, base=2.5)

        with self.assertRaises(TypeError):
            StateSpace([1.0, 2.0, 3.0])

    def test_invalid_spec_value(self):
        with self.assertRaises(ValueError):
            StateSpace(0)

        with self.assertRaises(ValueError):
            StateSpace(-1)

        with self.assertRaises(ValueError):
            StateSpace([])

        with self.assertRaises(ValueError):
            StateSpace([0])

        with self.assertRaises(ValueError):
            StateSpace([-1])

    def test_invalid_base_value(self):
        with self.assertRaises(ValueError):
            StateSpace(3, base=0)

        with self.assertRaises(ValueError):
            StateSpace(4, base=-1)

    def test_uniform_bases(self):
        spec = StateSpace(5)
        self.assertTrue(spec.is_uniform)
        self.assertEqual(5, spec.ndim)
        self.assertEqual(2, spec.base)
        self.assertEqual(32, spec.volume)

        spec = StateSpace(8, base=4)
        self.assertTrue(spec.is_uniform)
        self.assertEqual(8, spec.ndim)
        self.assertEqual(4, spec.base)
        self.assertEqual(65536, spec.volume)

        spec = StateSpace([3, 3, 3, 3])
        self.assertTrue(spec.is_uniform)
        self.assertEqual(4, spec.ndim)
        self.assertEqual(3, spec.base)
        self.assertEqual(81, spec.volume)

    def test_base_mismatch(self):
        with self.assertRaises(ValueError):
            StateSpace([2, 2, 2], base=3)

        with self.assertRaises(ValueError):
            StateSpace([3, 3, 3], base=2)

        with self.assertRaises(ValueError):
            StateSpace([2, 2, 3], base=2)

        with self.assertRaises(ValueError):
            StateSpace([2, 2, 3], base=3)

        StateSpace([2, 2, 2], base=2)
        StateSpace([3, 3, 3], base=3)

    def test_nonuniform_bases(self):
        spec = StateSpace([1, 2, 3, 2, 1])
        self.assertFalse(spec.is_uniform)
        self.assertEqual([1, 2, 3, 2, 1], spec.base)
        self.assertEqual(5, spec.ndim)
        self.assertEqual(12, spec.volume)

    def test_states_boolean(self):
        space = StateSpace(1)
        self.assertEqual([[0], [1]],
                         list(space))

        space = StateSpace(2)
        self.assertEqual([[0, 0], [1, 0], [0, 1], [1, 1]],
                         list(space))

        space = StateSpace(3)
        self.assertEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                         list(space))

    def test_states_nonboolean(self):
        space = StateSpace(1, base=1)
        self.assertEqual([[0]],
                         list(space))

        space = StateSpace(1, base=3)
        self.assertEqual([[0], [1], [2]],
                         list(space))

        space = StateSpace(2, base=1)
        self.assertEqual([[0, 0]],
                         list(space))

        space = StateSpace(2, base=3)
        self.assertEqual([[0, 0], [1, 0], [2, 0],
                          [0, 1], [1, 1], [2, 1],
                          [0, 2], [1, 2], [2, 2]],
                         list(space))

    def test_states_boolean_list(self):
        space = StateSpace([2])
        self.assertEqual([[0], [1]],
                         list(space))

        space = StateSpace([2, 2])
        self.assertEqual([[0, 0], [1, 0], [0, 1], [1, 1]],
                         list(space))

        space = StateSpace([2, 2, 2])
        self.assertEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                         list(space))

    def test_states_nonboolean_list(self):
        space = StateSpace([1])
        self.assertEqual([[0]],
                         list(space))

        space = StateSpace([3])
        self.assertEqual([[0], [1], [2]],
                         list(space))

        space = StateSpace([1, 2])
        self.assertEqual([[0, 0], [0, 1]],
                         list(space))

        space = StateSpace([1, 3])
        self.assertEqual([[0, 0], [0, 1], [0, 2]],
                         list(space))

        space = StateSpace([3, 3])
        self.assertEqual([[0, 0], [1, 0], [2, 0],
                          [0, 1], [1, 1], [2, 1],
                          [0, 2], [1, 2], [2, 2]],
                         list(space))

    def test_states_count(self):
        xs = [3, 5, 2, 5, 2, 1, 4, 2]
        space = StateSpace(xs)
        count = 0
        for state in space:
            count += 1
        self.assertEqual(np.product(xs), count)

    def test_encoding_error(self):
        space = StateSpace(3)
        with self.assertRaises(ValueError):
            space.encode([1, 1])

        space = StateSpace(1)
        with self.assertRaises(ValueError):
            space.encode([2])

        space = StateSpace([2, 3])
        with self.assertRaises(ValueError):
            space.encode([1, 3])

        with self.assertRaises(ValueError):
            space.encode([1, -1])

    def test_encoding_uniform(self):
        for width in range(1, 5):
            for base in range(1, 5):
                space = StateSpace(width, base)
                counter = 0
                for state in space:
                    encoding = space.encode(state)
                    self.assertEqual(counter, encoding)
                    counter += 1

    def test_encoding_nonuniform(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    space = StateSpace([a, b, c])
                    counter = 0
                    for state in space:
                        encoding = space.encode(state)
                        self.assertEqual(counter, encoding)
                        counter += 1

    def test_decoding_uniform(self):
        for width in range(1, 5):
            for base in range(1, 5):
                space = StateSpace(width, base)
                states = list(space)
                decoded = list(map(space.decode, range(space.volume)))
                self.assertEqual(states, decoded)

    def test_decoding_nonuniform(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    space = StateSpace([a, b, c])
                    states = list(space)
                    decoded = list(map(space.decode, range(space.volume)))
                    self.assertEqual(states, decoded)

    def test_encode_decode_uniform(self):
        for width in range(1, 5):
            for base in range(1, 5):
                space = StateSpace(width, base)
                for state in space:
                    encoded = space.encode(state)
                    decoded = space.decode(encoded)
                    self.assertEqual(state, decoded)

    def test_encode_decode_nonuniform(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    space = StateSpace([a, b, c])
                    for state in space:
                        encoded = space.encode(state)
                        decoded = space.decode(encoded)
                        self.assertEqual(state, decoded)

    def test_decode_encode_uniform(self):
        for width in range(1, 5):
            for base in range(1, 5):
                space = StateSpace(width, base)
                for i in range(base**width):
                    decoded = space.decode(i)
                    encoded = space.encode(decoded)
                    self.assertEqual(i, encoded)

    def test_decode_encode_nonuniform(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    space = StateSpace([a, b, c])
                    for i in range(a * b * c):
                        decoded = space.decode(i)
                        encoded = space.encode(decoded)
                        self.assertEqual(i, encoded)

    def test_check_states_uniform(self):
        state_space = StateSpace(3)
        self.assertTrue([0, 1, 1] in state_space)
        self.assertFalse([0, 0] in state_space)
        self.assertFalse([1, 2, 0] in state_space)

        self.assertFalse([0, 1, 1] not in state_space)
        self.assertTrue([0, 0] not in state_space)
        self.assertTrue([1, 2, 0] not in state_space)

    def test_check_states_varied(self):
        self.assertTrue([0, 2, 1] in StateSpace([2, 3, 2]))
        self.assertFalse([0, 1] in StateSpace([2, 2, 3]))
        self.assertFalse([1, 1, 6] in StateSpace([2, 3, 4]))

        self.assertFalse([0, 2, 1] not in StateSpace([2, 3, 2]))
        self.assertTrue([0, 1] not in StateSpace([2, 2, 3]))
        self.assertTrue([1, 1, 6] not in StateSpace([2, 3, 4]))

    def test_long_encoding(self):
        state_space = StateSpace(10)
        code = state_space.encode(np.ones(10, dtype=int))
        self.assertIsInstance(code, long)

        state_space = StateSpace(68)
        code = state_space.encode(np.ones(68, dtype=int))
        self.assertIsInstance(code, long)

        state_space = StateSpace(100)
        code = state_space.encode(np.ones(100, dtype=int))
        self.assertIsInstance(code, long)
