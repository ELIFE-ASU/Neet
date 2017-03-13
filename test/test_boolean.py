# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.boolean as bnet
import numpy as np

class TestWTNetwork(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import is_network
        self.assertTrue(is_network(bnet.WTNetwork))
        self.assertTrue(is_network(bnet.WTNetwork([[1]])))


    def test_is_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertTrue(is_fixed_sized(bnet.WTNetwork))
        self.assertTrue(is_fixed_sized(bnet.WTNetwork([[1]])))


    def test_init_failed(self):
        with self.assertRaises(ValueError):
            bnet.WTNetwork(None)

        with self.assertRaises(ValueError):
            bnet.WTNetwork('a')

        with self.assertRaises(ValueError):
            bnet.WTNetwork(0)

        with self.assertRaises(ValueError):
            bnet.WTNetwork(-1)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], 'a')

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], 0)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]], -1)

        with self.assertRaises(ValueError):
            bnet.WTNetwork([])
        
        with self.assertRaises(ValueError):
            bnet.WTNetwork([[]])
        
        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]],[])
        
        with self.assertRaises(ValueError):
            bnet.WTNetwork([[1]],[1,2])


    def test_init_weights(self):
        net = bnet.WTNetwork([[1]])
        self.assertEqual(1, net.size)
        self.assertTrue(np.array_equal([[1]], net.weights))
        self.assertTrue(np.array_equal([0], net.thresholds))

        net = bnet.WTNetwork([[1,0],[0,1]])
        self.assertEqual(2, net.size)
        self.assertTrue(np.array_equal([[1,0],[0,1]], net.weights))
        self.assertTrue(np.array_equal([0,0], net.thresholds))

        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(3, net.size)
        self.assertTrue(np.array_equal([[1,0,0],[0,1,0],[0,0,1]], net.weights))
        self.assertTrue(np.array_equal([0,0,0], net.thresholds))


    def test_init_weights_thresholds(self):
        net = bnet.WTNetwork([[1]], [1])
        self.assertEqual(1, net.size)
        self.assertTrue(np.array_equal([[1]], net.weights))
        self.assertTrue(np.array_equal([1], net.thresholds))

        net = bnet.WTNetwork([[1,0],[0,1]], [1,2])
        self.assertEqual(2, net.size)
        self.assertTrue(np.array_equal([[1,0],[0,1]], net.weights))
        self.assertTrue(np.array_equal([1,2], net.thresholds))

        net = bnet.WTNetwork([[1,0,0],[0,1,0],[0,0,1]], [1,2,3])
        self.assertEqual(3, net.size)
        self.assertTrue(np.array_equal([[1,0,0],[0,1,0],[0,0,1]], net.weights))
        self.assertTrue(np.array_equal([1,2,3], net.thresholds))
