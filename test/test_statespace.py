import unittest
from neet.python import long
from neet import StateSpace, UniformSpace
import numpy as np


class TestStateSpace(unittest.TestCase):
    def test_invalid_shape_type(self):
        with self.assertRaises(TypeError):
            StateSpace(2)

        with self.assertRaises(TypeError):
            StateSpace("a")

        with self.assertRaises(TypeError):
            StateSpace("abc")

        with self.assertRaises(TypeError):
            StateSpace([1.0, 2.0, 3.0])

    def test_invalid_shape_value(self):
        with self.assertRaises(ValueError):
            StateSpace([])

        with self.assertRaises(ValueError):
            StateSpace([0])

        with self.assertRaises(ValueError):
            StateSpace([-1])

    def test_uniform_shape(self):
        space = StateSpace([2, 2, 2, 2, 2])
        self.assertEqual(5, space.size)
        self.assertEqual([2, 2, 2, 2, 2], space.shape)
        self.assertEqual(32, space.volume)

        space = StateSpace([4, 4, 4, 4, 4, 4, 4, 4])
        self.assertEqual(8, space.size)
        self.assertEqual([4, 4, 4, 4, 4, 4, 4, 4], space.shape)
        self.assertEqual(65536, space.volume)

        space = StateSpace([3, 3, 3, 3])
        self.assertEqual(4, space.size)
        self.assertEqual([3, 3, 3, 3], space.shape)
        self.assertEqual(81, space.volume)

    def test_nonuniform_shape(self):
        space = StateSpace([1, 2, 3, 2, 1])
        self.assertEqual(5, space.size)
        self.assertEqual([1, 2, 3, 2, 1], space.shape)
        self.assertEqual(12, space.volume)

    def test_states_boolean(self):
        space = StateSpace([2])
        self.assertEqual([[0], [1]], list(space))

        space = StateSpace([2, 2])
        self.assertEqual([[0, 0], [1, 0], [0, 1], [1, 1]], list(space))

        space = StateSpace([2, 2, 2])
        self.assertEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                         list(space))

    def test_states_nonboolean(self):
        space = StateSpace([1])
        self.assertEqual([[0]],
                         list(space))

        space = StateSpace([3])
        self.assertEqual([[0], [1], [2]],
                         list(space))

        space = StateSpace([1, 1])
        self.assertEqual([[0, 0]],
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
        space = StateSpace([2, 2, 2])
        with self.assertRaises(ValueError):
            space.encode([1, 1])

        space = StateSpace([2])
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
                space = StateSpace([base] * width)
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
                space = StateSpace([base] * width)
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
                space = StateSpace([base] * width)
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
                space = StateSpace([base] * width)
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
        state_space = StateSpace([2, 2, 2])
        self.assertTrue([0, 1, 1] in state_space)
        self.assertFalse([0, 0] in state_space)
        self.assertFalse([1, 2, 0] in state_space)

        self.assertFalse([0, 1, 1] not in state_space)
        self.assertTrue([0, 0] not in state_space)
        self.assertTrue([1, 2, 0] not in state_space)

        self.assertFalse(1 in state_space)
        self.assertFalse("string" in state_space)

    def test_check_states_varied(self):
        self.assertTrue([0, 2, 1] in StateSpace([2, 3, 2]))
        self.assertFalse([0, 1] in StateSpace([2, 2, 3]))
        self.assertFalse([1, 1, 6] in StateSpace([2, 3, 4]))

        self.assertFalse([0, 2, 1] not in StateSpace([2, 3, 2]))
        self.assertTrue([0, 1] not in StateSpace([2, 2, 3]))
        self.assertTrue([1, 1, 6] not in StateSpace([2, 3, 4]))

        self.assertFalse(1 in StateSpace([2, 3, 2]))
        self.assertFalse("string" in StateSpace([2, 3, 2]))

    def test_long_encoding(self):
        state_space = StateSpace([2] * 10)
        code = state_space.encode(np.ones(10, dtype=int))
        self.assertIsInstance(code, long)

        state_space = StateSpace([2] * 68)
        code = state_space.encode(np.ones(68, dtype=int))
        self.assertIsInstance(code, long)

        state_space = StateSpace([2] * 100)
        code = state_space.encode(np.ones(100, dtype=int))
        self.assertIsInstance(code, long)


class TestUniformSpace(unittest.TestCase):
    def test_invalid_size_type(self):
        with self.assertRaises(TypeError):
            UniformSpace('a', 2)

        with self.assertRaises(TypeError):
            UniformSpace([2], 2)

        with self.assertRaises(TypeError):
            UniformSpace(3.0, 2)

    def test_invalid_shape_value(self):
        with self.assertRaises(ValueError):
            UniformSpace(0, 2)

        with self.assertRaises(ValueError):
            UniformSpace(-1, 2)

    def test_invalid_base_type(self):
        with self.assertRaises(TypeError):
            UniformSpace(5, 'a')

        with self.assertRaises(TypeError):
            UniformSpace(5, [2])

        with self.assertRaises(TypeError):
            UniformSpace(5, 3.0)

    def test_invalid_base_value(self):
        with self.assertRaises(ValueError):
            UniformSpace(5, 0)

        with self.assertRaises(ValueError):
            UniformSpace(5, -1)

    def test_shape(self):
        space = UniformSpace(size=5, base=2)
        self.assertEqual(2, space.base)
        self.assertEqual(32, space.volume)
        self.assertEqual(5, space.size)
        self.assertEqual([2, 2, 2, 2, 2], space.shape)

        space = UniformSpace(size=8, base=4)
        self.assertEqual(4, space.base)
        self.assertEqual(65536, space.volume)
        self.assertEqual(8, space.size)
        self.assertEqual([4, 4, 4, 4, 4, 4, 4, 4], space.shape)

    def test_states_boolean(self):
        space = UniformSpace(size=1, base=2)
        self.assertEqual([[0], [1]], list(space))

        space = UniformSpace(size=2, base=2)
        self.assertEqual([[0, 0], [1, 0], [0, 1], [1, 1]], list(space))

        space = UniformSpace(size=3, base=2)
        self.assertEqual([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                         list(space))

    def test_states_nonboolean(self):
        space = UniformSpace(size=1, base=1)
        self.assertEqual([[0]], list(space))

        space = UniformSpace(size=1, base=3)
        self.assertEqual([[0], [1], [2]], list(space))

        space = UniformSpace(size=2, base=1)
        self.assertEqual([[0, 0]], list(space))

        space = UniformSpace(size=2, base=3)
        self.assertEqual([[0, 0], [1, 0], [2, 0],
                          [0, 1], [1, 1], [2, 1],
                          [0, 2], [1, 2], [2, 2]],
                         list(space))

    def test_states_count(self):
        space = UniformSpace(size=5, base=3)
        count = 0
        for state in space:
            count += 1
        self.assertEqual(3**5, count)

    def test_encoding_error(self):
        space = UniformSpace(size=3, base=2)
        with self.assertRaises(ValueError):
            space.encode([1, 1])

        space = UniformSpace(size=1, base=2)
        with self.assertRaises(ValueError):
            space.encode([2])

        space = UniformSpace(size=2, base=3)
        with self.assertRaises(ValueError):
            space.encode([1, 3])

        with self.assertRaises(ValueError):
            space.encode([1, -1])

    def test_encoding(self):
        for size in range(1, 5):
            for base in range(1, 5):
                space = UniformSpace(size, base)
                counter = 0
                for state in space:
                    encoding = space.encode(state)
                    self.assertEqual(counter, encoding)
                    counter += 1

    def test_decoding(self):
        for size in range(1, 5):
            for base in range(1, 5):
                space = UniformSpace(size, base)
                states = list(space)
                decoded = list(map(space.decode, range(space.volume)))
                self.assertEqual(states, decoded)

    def test_encode_decode(self):
        for size in range(1, 5):
            for base in range(1, 5):
                space = UniformSpace(size, base)
                for state in space:
                    encoded = space.encode(state)
                    decoded = space.decode(encoded)
                    self.assertEqual(state, decoded)

    def test_decode_encode(self):
        for size in range(1, 5):
            for base in range(1, 5):
                space = UniformSpace(size, base)
                for i in range(base**size):
                    decoded = space.decode(i)
                    encoded = space.encode(decoded)
                    self.assertEqual(i, encoded)

    def test_check_states(self):
        state_space = UniformSpace(size=3, base=2)
        self.assertTrue([0, 1, 1] in state_space)
        self.assertFalse([0, 0] in state_space)
        self.assertFalse([1, 2, 0] in state_space)

        self.assertFalse([0, 1, 1] not in state_space)
        self.assertTrue([0, 0] not in state_space)
        self.assertTrue([1, 2, 0] not in state_space)

        self.assertFalse(1 in state_space)
        self.assertFalse("string" in state_space)

        state_space = UniformSpace(size=3, base=3)
        self.assertTrue([0, 1, 1] in state_space)
        self.assertFalse([0, 0] in state_space)
        self.assertTrue([1, 2, 0] in state_space)
        self.assertFalse([1, 2, 3] in state_space)

        self.assertFalse([0, 1, 1] not in state_space)
        self.assertTrue([0, 0] not in state_space)
        self.assertFalse([1, 2, 0] not in state_space)
        self.assertTrue([1, 2, 3] not in state_space)

        self.assertFalse(1 in state_space)
        self.assertFalse("string" in state_space)

    def test_long_encoding(self):
        state_space = UniformSpace(size=10, base=2)
        code = state_space.encode(np.ones(10, dtype=int))
        self.assertIsInstance(code, long)

        state_space = UniformSpace(size=68, base=2)
        code = state_space.encode(np.ones(68, dtype=int))
        self.assertIsInstance(code, long)

        state_space = UniformSpace(size=100, base=2)
        code = state_space.encode(np.ones(100, dtype=int))
        self.assertIsInstance(code, long)
