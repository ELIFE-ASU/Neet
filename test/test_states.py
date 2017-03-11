# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet
import numpy as np

class TestStateSpace(unittest.TestCase):
    def test_invalid_spec_type(self):
        with self.assertRaises(TypeError):
            neet.StateSpace("a")

        with self.assertRaises(TypeError):
            neet.StateSpace("abc")

    def test_invalid_base_type(self):
        with self.assertRaises(TypeError):
            neet.StateSpace(5, b='a')

        with self.assertRaises(TypeError):
            neet.StateSpace(3, b=2.5)

        with self.assertRaises(TypeError):
            neet.StateSpace([1.0, 2.0, 3.0])

    def test_invalid_spec_value(self):
        with self.assertRaises(ValueError):
            neet.StateSpace(0)

        with self.assertRaises(ValueError):
            neet.StateSpace(-1)

        with self.assertRaises(ValueError):
            neet.StateSpace([])

        with self.assertRaises(ValueError):
            neet.StateSpace([0])

        with self.assertRaises(ValueError):
            neet.StateSpace([-1])

    def test_invalid_base_value(self):
        with self.assertRaises(ValueError):
            neet.StateSpace(3, b=0)

        with self.assertRaises(ValueError):
            neet.StateSpace(4, b=-1)

    def test_uniform_bases(self):
        spec = neet.StateSpace(5)
        self.assertTrue(spec.is_uniform)
        self.assertEqual(5, spec.ndim)
        self.assertEqual(2, spec.base)

        spec = neet.StateSpace(8, b=4)
        self.assertTrue(spec.is_uniform)
        self.assertEqual(8, spec.ndim)
        self.assertEqual(4, spec.base)

        spec = neet.StateSpace([3,3,3,3])
        self.assertTrue(spec.is_uniform)
        self.assertEqual(4, spec.ndim)
        self.assertEqual(3, spec.base)

    def test_uniform_bases(self):
        spec = neet.StateSpace([1,2,3,2,1])
        self.assertFalse(spec.is_uniform)
        self.assertEqual([1,2,3,2,1], spec.bases)
        self.assertEqual(5, spec.ndim)

    def test_states_boolean(self):
        space = neet.StateSpace(1)
        self.assertEqual([[0],[1]],
            list(space.states()))

        space = neet.StateSpace(2)
        self.assertEqual([[0,0],[1,0],[0,1],[1,1]],
            list(space.states()))

        space = neet.StateSpace(3)
        self.assertEqual([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                          [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
            list(space.states()))

    def test_states_nonboolean(self):
        space = neet.StateSpace(1, b=1)
        self.assertEqual([[0]],
            list(space.states()))

        space = neet.StateSpace(1, b=3)
        self.assertEqual([[0],[1],[2]],
            list(space.states()))

        space = neet.StateSpace(2, b=1)
        self.assertEqual([[0,0]],
            list(space.states()))

        space = neet.StateSpace(2, b=3)
        self.assertEqual([[0,0],[1,0],[2,0],
                          [0,1],[1,1],[2,1],
                          [0,2],[1,2],[2,2]],
            list(space.states()))

    def test_states_boolean_list(self):
        space = neet.StateSpace([2])
        self.assertEqual([[0],[1]],
            list(space.states()))

        space = neet.StateSpace([2,2])
        self.assertEqual([[0,0],[1,0],[0,1],[1,1]],
            list(space.states()))

        space = neet.StateSpace([2,2,2])
        self.assertEqual([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                          [0,0,1],[1,0,1],[0,1,1],[1,1,1]],
            list(space.states()))

    def test_states_nonboolean_list(self):
        space = neet.StateSpace([1])
        self.assertEqual([[0]],
            list(space.states()))

        space = neet.StateSpace([3])
        self.assertEqual([[0],[1],[2]],
            list(space.states()))

        space = neet.StateSpace([1,2])
        self.assertEqual([[0,0],[0,1]],
            list(space.states()))

        space = neet.StateSpace([1,3])
        self.assertEqual([[0,0],[0,1],[0,2]],
            list(space.states()))

        space = neet.StateSpace([3,3])
        self.assertEqual([[0,0],[1,0],[2,0],
                          [0,1],[1,1],[2,1],
                          [0,2],[1,2],[2,2]],
            list(space.states()))

    def test_states_count(self):
        xs = [3,5,2,5,2,1,4,2]
        space = neet.StateSpace(xs)
        count = 0;
        for state in space.states():
            count += 1
        self.assertEqual(np.product(xs), count)

    def test_encoding_error(self):
        space = neet.StateSpace(3)
        with self.assertRaises(ValueError):
            space.encode([1,1])

        space = neet.StateSpace(1)
        with self.assertRaises(ValueError):
            space.encode([2])

        space = neet.StateSpace([2,3])
        with self.assertRaises(ValueError):
            space.encode([1,3])

        with self.assertRaises(ValueError):
            space.encode([1,-1])

    def test_encoding_uniform(self):
        for width in range(1,5):
            for base in range(1,5):
                space = neet.StateSpace(width, base)
                counter = 0
                for state in space.states():
                    encoding = space.encode(state)
                    self.assertEqual(counter, encoding)
                    counter += 1

    def test_encoding_nonuniform(self):
        for a in range(1,5):
            for b in range(1,5):
                for c in range(1,5):
                    space = neet.StateSpace([a,b,c])
                    counter = 0
                    for state in space.states():
                        encoding = space.encode(state)
                        self.assertEqual(counter, encoding)
                        counter += 1

    def test_decoding_uniform(self):
        for width in range(1,5):
            for base in range(1,5):
                space = neet.StateSpace(width, base)
                states = list(space.states())
                decoded = list(map(space.decode, range(0, base**width)))
                self.assertEqual(states, decoded)

    def test_decoding_uniform(self):
        for a in range(1,5):
            for b in range(1,5):
                for c in range(1,5):
                    space = neet.StateSpace([a,b,c])
                    states = list(space.states())
                    decoded = list(map(space.decode, range(0, a*b*c)))
                    self.assertEqual(states, decoded)

    def test_encode_decode_uniform(self):
        for width in range(1,5):
            for base in range(1,5):
                space = neet.StateSpace(width, base)
                for state in space.states():
                    encoded = space.encode(state)
                    decoded = space.decode(encoded)
                    self.assertEqual(state, decoded)

    def test_encode_decode_nonuniform(self):
        for a in range(1,5):
            for b in range(1,5):
                for c in range(1,5):
                    space = neet.StateSpace([a,b,c])
                    for state in space.states():
                        encoded = space.encode(state)
                        decoded = space.decode(encoded)
                        self.assertEqual(state, decoded)

    def test_decode_encode_uniform(self):
        for width in range(1,5):
            for base in range(1,5):
                space = neet.StateSpace(width, base)
                for i in range(base**width):
                    decoded = space.decode(i)
                    encoded = space.encode(decoded)
                    self.assertEqual(i, encoded)

    def test_decode_encode_nonuniform(self):
        for a in range(1,5):
            for b in range(1,5):
                for c in range(1,5):
                    space = neet.StateSpace([a,b,c])
                    for i in range(a*b*c):
                        decoded = space.decode(i)
                        encoded = space.encode(decoded)
                        self.assertEqual(i, encoded)
