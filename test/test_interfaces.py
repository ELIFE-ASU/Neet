# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.interfaces import *
import numpy as np

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return StateSpace(1)

    class FixedSizeNetwork(IsNetwork):
        def size(self):
            return 5

    class IsNotNetwork(object):
        pass

    class NotFixedSizedNetwork(IsNotNetwork):
        def size(self):
            return 5

    def test_is_network(self):
        net = self.IsNetwork()
        self.assertTrue(is_network(net))
        self.assertTrue(is_network(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(is_network(not_net))
        self.assertFalse(is_network(type(not_net)))

        self.assertFalse(is_network(5))
        self.assertFalse(is_network(int))

    def test_is_fixed_sized(self):
        net = self.IsNetwork()
        self.assertFalse(is_fixed_sized(net))
        self.assertFalse(is_fixed_sized(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(is_fixed_sized(not_net))
        self.assertFalse(is_fixed_sized(type(not_net)))

        net = self.FixedSizeNetwork()
        self.assertTrue(is_fixed_sized(net))
        self.assertTrue(is_fixed_sized(type(net)))

        not_net = self.NotFixedSizedNetwork()
        self.assertFalse(is_fixed_sized(not_net))
        self.assertFalse(is_fixed_sized(type(not_net)))
