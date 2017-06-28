# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.automata as ca
import numpy as np

class TestECA(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import is_network
        self.assertTrue(is_network(ca.ECA))
        self.assertTrue(is_network(ca.ECA(23)))


    def test_is_not_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertFalse(is_fixed_sized(ca.ECA))
        self.assertFalse(is_fixed_sized(ca.ECA(23)))


    def test_fail_init(self):
        with self.assertRaises(ValueError):
            ca.ECA(-1)

        with self.assertRaises(ValueError):
            ca.ECA(256)

        with self.assertRaises(TypeError):
            ca.ECA([1,1,0,1,1,0,0,1])

        with self.assertRaises(TypeError):
            ca.ECA("30")

        with self.assertRaises(TypeError):
            ca.ECA(30, boundary=[1,2])

        with self.assertRaises(ValueError):
            ca.ECA(30, boundary=(1,0,1))

        with self.assertRaises(ValueError):
            ca.ECA(30, boundary=(1,2))


    def test_init(self):
        for code in range(256):
            for left in range(2):
                for right in range(2):
                    eca = ca.ECA(code, (left,right))
                    self.assertEqual(code, eca.code)
                    self.assertEqual((left,right), eca.boundary)


    def test_invalid_code(self):
        eca = ca.ECA(30)

        eca.code = 45

        with self.assertRaises(ValueError):
            eca.code = -1

        with self.assertRaises(ValueError):
            eca.code = 256

        with self.assertRaises(TypeError):
            eca.code = "30"


    def test_invalid_boundary(self):
        eca = ca.ECA(30)

        eca.boundary = (0,0)
        eca.boundary = None

        with self.assertRaises(ValueError):
            eca.boundary = (1,1,1)

        with self.assertRaises(ValueError):
            eca.boundary = (1,2)

        with self.assertRaises(TypeError):
            eca.boundary = 1

        with self.assertRaises(TypeError):
            eca.boundary = [0,1]

    def test_state_space(self):
        with self.assertRaises(ValueError):
            ca.ECA(30).state_space(0)

        with self.assertRaises(ValueError):
            ca.ECA(30).state_space(-1)

        eca = ca.ECA(30)
        self.assertEqual(2, len(list(eca.state_space(1))))
        self.assertEqual(4, len(list(eca.state_space(2))))
        self.assertEqual(8, len(list(eca.state_space(3))))


    def test_lattice_empty_update(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
            eca.update([])


    def test_invalid_lattice_state_update(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
            eca.update([-1,0,1])

        with self.assertRaises(ValueError):
            eca.update([1,0,-1])

        with self.assertRaises(ValueError):
            eca.update([2,0,0])

        with self.assertRaises(ValueError):
            eca.update([1,0,2])

        with self.assertRaises(ValueError):
            eca.update([[1],[0],[2]])

        with self.assertRaises(ValueError):
            eca.update("101")


    def test_update_closed(self):
        eca = ca.ECA(30)

        lattice = [0]

        eca.update(lattice)
        self.assertEqual([0], lattice)

        lattice = [0,0]

        eca.update(lattice)
        self.assertEqual([0,0], lattice)

        lattice = [0,0,1,0,0]

        eca.update(lattice)
        self.assertEqual([0,1,1,1,0], lattice)

        eca.update(lattice)
        self.assertEqual([1,1,0,0,1], lattice)


    def test_update_open(self):
        eca = ca.ECA(30, (0,1))

        lattice = [0]

        eca.update(lattice)
        self.assertEqual([1], lattice)

        lattice = [0,0]

        eca.update(lattice)
        self.assertEqual([0,1], lattice)

        lattice = [0,0,1,0,0]

        eca.update(lattice)
        self.assertEqual([0,1,1,1,1], lattice)

        eca.update(lattice)
        self.assertEqual([1,1,0,0,0], lattice)


    def test_update_long_time_closed(self):
        eca = ca.ECA(45)
        lattice  = [1,1,0,1,0,0,1,0,1,0,0,1,0,1]
        expected = [0,1,1,0,1,0,1,0,1,0,1,0,1,0]
        state_space = eca.state_space(len(lattice))
        if lattice in state_space:
            for n in range(1000):
                eca._unsafe_update(lattice)
        self.assertEqual(expected, lattice)


    def test_update_long_time_open(self):
        eca = ca.ECA(45, (0,1))
        lattice  = [1,1,0,1,0,0,1,0,1,0,0,1,0,1]
        expected = [1,0,0,1,0,0,1,0,0,1,0,0,1,1]
        state_space = eca.state_space(len(lattice))
        if lattice in state_space:
            for n in range(1000):
                eca._unsafe_update(lattice)
        self.assertEqual(expected, lattice)


    def test_update_numpy(self):
        eca = ca.ECA(30, (0,1))

        lattice = np.asarray([0])

        eca.update(lattice)
        self.assertTrue(np.array_equal([1], lattice))

        lattice = np.asarray([0,0])

        eca.update(lattice)
        self.assertTrue(np.array_equal([0,1], lattice))

        lattice = [0,0,1,0,0]

        eca.update(lattice)
        self.assertTrue(np.array_equal([0,1,1,1,1], lattice))

        eca.update(lattice)
        self.assertTrue(np.array_equal([1,1,0,0,0], lattice))


    def test_update_index_error(self):
        eca = ca.ECA(30)
        with self.assertRaises(IndexError):
            eca.update([0,0], index=2)

        with self.assertRaises(IndexError):
            eca.update([0,0], index=-3)


    def test_update_index(self):
        eca = ca.ECA(30, (1,1))

        lattice = [0,0,0,0,0]
        eca.update(lattice, index=0)
        self.assertEqual([1,0,0,0,0], lattice)

        lattice = [0,0,0,0,0]
        eca.update(lattice, index=1)
        self.assertEqual([0,0,0,0,0], lattice)

        lattice = [0,0,0,0,0]
        eca.update(lattice, index=-1)
        self.assertEqual([0,0,0,0,1], lattice)

        lattice = [0,0,1,0,0]
        eca.update(lattice, index=1)
        self.assertEqual([0,1,1,0,0], lattice)


    def test_update_index_numpy(self):
        eca = ca.ECA(30, (1,1))

        lattice = np.asarray([0,0,0,0,0])
        eca.update(lattice, index=0)
        self.assertTrue(np.array_equal([1,0,0,0,0], lattice))

        lattice = np.asarray([0,0,0,0,0])
        eca.update(lattice, index=1)
        self.assertTrue(np.array_equal([0,0,0,0,0], lattice))

        lattice = np.asarray([0,0,0,0,0])
        eca.update(lattice, index=-1)
        self.assertTrue(np.array_equal([0,0,0,0,1], lattice))

        lattice = np.asarray([0,0,1,0,0])
        eca.update(lattice, index=1)
        self.assertTrue(np.array_equal([0,1,1,0,0], lattice))
