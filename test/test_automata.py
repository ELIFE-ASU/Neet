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


    def test_update_pin_none(self):
        eca = ca.ECA(30)

        xs = [0,0,1,0,0]
        self.assertEqual([0,1,1,1,0], eca.update(xs, pin=None))
        self.assertEqual([1,1,0,0,1], eca.update(xs, pin=[]))


    def test_update_pin_index_clash(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
          eca.update([0,0], index=0, pin=[1])
        with self.assertRaises(ValueError):
          eca.update([0,0], index=1, pin=[1])
        with self.assertRaises(ValueError):
          eca.update([0,0], index=1, pin=[0,1])


    def test_update_pin(self):
        eca = ca.ECA(30)

        xs = [0,0,1,0,0]
        self.assertEqual([0,0,1,1,0], eca.update(xs, pin=[1]))
        self.assertEqual([0,0,1,0,1], eca.update(xs, pin=[1]))
        self.assertEqual([1,0,1,0,1], eca.update(xs, pin=[1]))

        eca.boundary = (1,1)
        xs = [0,0,0,0,0]
        self.assertEqual([1,0,0,0,0], eca.update(xs, pin=[-1]))
        self.assertEqual([1,1,0,0,0], eca.update(xs, pin=[0,-1]))


    def test_update_values_none(self):
        eca = ca.ECA(30)

        xs = [0,0,1,0,0]
        self.assertEqual([0,1,1,1,0], eca.update(xs, values=None))
        self.assertEqual([1,1,0,0,1], eca.update(xs, values={}))


    def test_update_invalid_values(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], values={0: 2})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], values={0:-1})


    def test_update_values_index_clash(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], index=0, values={0: 1})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], index=1, values={1: 0})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], index=1, values={0: 0, 1: 0})


    def test_update_values_pin_clash(self):
        eca = ca.ECA(30)
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], pin=[0], values={0: 1})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], pin=[1], values={1: 0})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], pin=[1], values={0: 0, 1: 0})
        with self.assertRaises(ValueError):
          eca.update([0,0,0,0,0], pin=[1, 0], values={0: 0})


    def test_update_values(self):
        eca = ca.ECA(30)

        xs = [0,0,1,0,0]
        self.assertEqual([0,1,0,1,0], eca.update(xs, values={2:0}))
        self.assertEqual([1,0,0,0,1], eca.update(xs, values={1:0, 3:0}))
        self.assertEqual([0,1,0,1,0], eca.update(xs, values={-1:0}))

        eca.boundary = (1,1)
        xs = [0,0,0,0,0]
        self.assertEqual([1,0,1,0,1], eca.update(xs, values={2:1}))

    def test_neighbors(self):

        net = ca.ECA(30)

        self.assertEqual(net.neighbors(3,index=2,direction='in'),set([0,1,2]))
        self.assertEqual(net.neighbors(3,direction='in'),[set([0, 1, 2]),  
                                                        set([0, 1, 2]), 
                                                        set([0, 1, 2])])
        self.assertEqual(net.neighbors(3,index=2,direction='out'),set([0,1,2]))
        self.assertEqual(net.neighbors(4,direction='out'),[set([0, 1, 3]), 
                                                         set([0,1,2]), 
                                                         set([1,2,3]), 
                                                         set([0,2,3])])

        self.assertEqual(net.neighbors(4,direction='both'),[set([0, 1, 3]), 
                                                         set([0,1,2]), 
                                                         set([1,2,3]), 
                                                         set([0,2,3])])

        self.assertEqual(net.neighbors(4,index=2,direction='both'),set([1,2,3]))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=3,direction='in'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=3,direction='out'))

        ## Add fixed-boundary
        net.boundary = (1,1)

        self.assertEqual(net.neighbors(3,index=2,direction='in'),set([1,2,3]))
        self.assertEqual(net.neighbors(3,index=3,direction='in'),set([]))

        self.assertEqual(net.neighbors(3,index=2,direction='out'),set([1,2]))
        self.assertEqual(net.neighbors(3,index=3,direction='out'),set([2]))
        self.assertEqual(net.neighbors(3,index=4,direction='out'),set([0]))

        self.assertEqual(net.neighbors(3,index=4,direction='both'),set([0]))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=5,direction='in'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=5,direction='out'))

        with self.assertRaises(TypeError):
            self.assertEqual(net.neighbors(3,index='2',direction='in'))

        with self.assertRaises(TypeError):
            self.assertEqual(net.neighbors(3,index='2',direction='out'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=-1,direction='in'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(3,index=-1,direction='out'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(0,index=1,direction='in'))

        with self.assertRaises(ValueError):
            self.assertEqual(net.neighbors(0,index=1,direction='out'))
