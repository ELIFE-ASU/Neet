# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""Unit test for LogicNetwork"""
import unittest
import neet.boolean as bnet


class TestLogicNetwork(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import is_network
        self.assertTrue(is_network(bnet.LogicNetwork([([0], {'0'})])))

    def test_is_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertTrue(is_fixed_sized(bnet.LogicNetwork([([0], {'0'})])))

    def test_init(self):
        net = bnet.LogicNetwork([((0,), {'0'})])
        self.assertEqual(1, net.size)
        self.assertEqual([(1, {0})], net._encoded_table)

        net = bnet.LogicNetwork(
            [((1,), {'0', '1'}), ((0,), {'1'})], ['A', 'B'])
        self.assertEqual(2, net.size)
        self.assertEqual(['A', 'B'], net.names)
        self.assertEqual([(2, {0, 2}), (1, {1})], net._encoded_table)

    def test_update(self):
        net = bnet.LogicNetwork([((0,), {'0'})])
        self.assertEqual(net.update([0], 0), [1])
        self.assertEqual(net.update([1], 0), [0])
        self.assertEqual(net.update([0]), [1])
        self.assertEqual(net.update([1]), [0])

        net = bnet.LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        self.assertEqual(net.update([0, 0], 0), [1, 0])
        self.assertEqual(net.update([0, 0], 1), [0, 0])
        self.assertEqual(net.update([0, 0]), [1, 0])
        self.assertEqual(net.update([0, 1], 0), [1, 1])
        self.assertEqual(net.update([0, 1], 1), [0, 0])
        self.assertEqual(net.update([0, 1]), [1, 0])
        self.assertEqual(net.update([1, 0], 0), [1, 0])
        self.assertEqual(net.update([1, 0], 1), [1, 1])
        self.assertEqual(net.update([1, 0]), [1, 1])
        self.assertEqual(net.update([1, 1]), [1, 1])

        net = bnet.LogicNetwork([((1, 2), {'01', '10'}),
                                 ((0, 2), {(0, 1), '10', (1, 1)}),
                                 ((0, 1), {'11'})])
        self.assertEqual(net.update([0, 1, 0]), [1, 0, 0])
        self.assertEqual(net.update([1, 1, 1], 1), [1, 1, 1])
        self.assertEqual(net.update([0, 0, 1]), [1, 1, 0])
        self.assertEqual(net.update([0, 0, 1], pin=[1]), [1, 0, 0])
        self.assertEqual(net.update([0, 0, 1], pin=[0, 1]), [0, 0, 0])
