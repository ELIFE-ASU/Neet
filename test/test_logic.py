# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""Unit test for LogicNetwork"""
import unittest
import neet.boolean as bnet


class TestLogicNetwork(unittest.TestCase):
    def test_is_network(self):
        from neet.interfaces import is_network
        self.assertTrue(is_network(bnet.LogicNetwork))
        self.assertTrue(is_network(bnet.LogicNetwork([(1, {0})])))

    def test_is_fixed_sized(self):
        from neet.interfaces import is_fixed_sized
        self.assertTrue(is_fixed_sized(bnet.LogicNetwork))
        self.assertTrue(is_fixed_sized(bnet.LogicNetwork([(1, {0})])))

    def test_init_failed(self):
        with self.assertRaises(TypeError):
            bnet.LogicNetwork(None)

        with self.assertRaises(TypeError):
            bnet.LogicNetwork({})

        with self.assertRaises(TypeError):
            bnet.LogicNetwork(1)

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([{}])

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([(), (), ()])

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([("1", {0})])

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([(1, (0))])

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([(2, {0})])

        with self.assertRaises(ValueError):
            bnet.LogicNetwork([(1, {2})])
