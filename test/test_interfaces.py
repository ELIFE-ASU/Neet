# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet
import numpy as np

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return neet.StateSpace(1)

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
        self.assertTrue(neet.is_network(net))
        self.assertTrue(neet.is_network(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_network(not_net))
        self.assertFalse(neet.is_network(type(not_net)))

        self.assertFalse(neet.is_network(5))
        self.assertFalse(neet.is_network(int))

    def test_is_fixed_sized(self):
        net = self.IsNetwork()
        self.assertFalse(neet.is_fixed_sized(net))
        self.assertFalse(neet.is_fixed_sized(type(net)))

        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_fixed_sized(not_net))
        self.assertFalse(neet.is_fixed_sized(type(not_net)))

        net = self.FixedSizeNetwork()
        self.assertTrue(neet.is_fixed_sized(net))
        self.assertTrue(neet.is_fixed_sized(type(net)))

        not_net = self.NotFixedSizedNetwork()
        self.assertFalse(neet.is_fixed_sized(not_net))
        self.assertFalse(neet.is_fixed_sized(type(not_net)))
