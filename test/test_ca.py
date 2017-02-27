# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.ca as ca

class TestECA(unittest.TestCase):
    def test_fail_init(self):
        with self.assertRaises(ValueError):
            ca.ECA(-1)

        with self.assertRaises(ValueError):
            ca.ECA(256)

        with self.assertRaises(TypeError):
            ca.ECA([1,1,0,1,1,0,0,1])

        with self.assertRaises(TypeError):
            ca.ECA("30")


    def test_init(self):
        for code in range(256):
            eca = ca.ECA(code)
            self.assertEqual(code, eca.code)


    def test_invalid_update_duration(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
            eca.update([0,0,0], 0)

        with self.assertRaises(ValueError):
            eca.update([0,0,0], -1)

        with self.assertRaises(TypeError):
            eca.update([0,0,0], [5])

        with self.assertRaises(TypeError):
            eca.update([0,0,0], "apples")


    def test_lattice_too_short_update(self):
        eca = ca.ECA(30)
        for width in range(3):
            with self.assertRaises(ValueError):
                eca.update([0]*width)


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


    def test_update(self):
        eca = ca.ECA(30)

        lattice = [0,0,1,0,0]

        eca.update(lattice)
        self.assertEqual([0,1,1,1,0], lattice)

        eca.update(lattice)
        self.assertEqual([1,1,0,0,1], lattice)

        eca.update(lattice, n=2)
        self.assertEqual([1,1,1,0,0], lattice)


    def test_update_long_time(self):
        eca = ca.ECA(45)
        lattice = [1,1,0,1,0,0,1,0,1,0,0,1,0,1]
        got = [0,1,1,0,1,0,1,0,1,0,1,0,1,0]
        eca.update(lattice, n=1000)
        self.assertEqual(got, lattice)


    def test_invalid_step_duration(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
            eca.step([0,0,0], 0)

        with self.assertRaises(ValueError):
            eca.step([0,0,0], -1)

        with self.assertRaises(TypeError):
            eca.step([0,0,0], [5])

        with self.assertRaises(TypeError):
            eca.step([0,0,0], "apples")


    def test_lattice_too_short_step(self):
        eca = ca.ECA(30)
        for width in range(3):
            with self.assertRaises(ValueError):
                eca.step([0]*width)


    def test_invalid_lattice_state_step(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
            eca.step([-1,0,1])

        with self.assertRaises(ValueError):
            eca.step([1,0,-1])

        with self.assertRaises(ValueError):
            eca.step([2,0,0])

        with self.assertRaises(ValueError):
            eca.step([1,0,2])

        with self.assertRaises(ValueError):
            eca.update([[1],[0],[2]])

        with self.assertRaises(ValueError):
            eca.update("101")


    def test_step(self):
        eca = ca.ECA(30)

        table = {1: [0,1,1,1,0],
                2: [1,1,0,0,1],
                4: [1,1,1,0,0]}

        lattice = [0,0,1,0,0]

        for n in table.keys():
            got = eca.step(lattice, n=n)
            self.assertEqual(table[n], got)
            self.assertEqual([0,0,1,0,0], lattice)


    def test_step_long_time(self):
        eca = ca.ECA(45)
        lattice  = [1,1,0,1,0,0,1,0,1,0,0,1,0,1]
        expected = [0,1,1,0,1,0,1,0,1,0,1,0,1,0]
        got = eca.step(lattice, n=1000)
        self.assertEqual(expected, got)

