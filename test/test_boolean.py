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
        self.assertTrue(is_network(bnet.WTNetwork(5)))


    def test_is_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertTrue(is_fixed_sized(bnet.WTNetwork))
        self.assertTrue(is_fixed_sized(bnet.WTNetwork(23)))

    def test_init_failed(self):
        with self.assertRaises(TypeError):
            bnet.WTNetwork(None)

        with self.assertRaises(TypeError):
            bnet.WTNetwork('a')

        with self.assertRaises(ValueError):
            bnet.WTNetwork(0)

        with self.assertRaises(ValueError):
            bnet.WTNetwork(-1)

    def test_init(self):
        net = bnet.WTNetwork(1)
        self.assertEqual(1, net.size)

        net = bnet.WTNetwork(2)
        self.assertEqual(2, net.size)

        net = bnet.WTNetwork(3)
        self.assertEqual(3, net.size)
