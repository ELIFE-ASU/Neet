# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from neet.boolean.random_network import *
from neet.boolean import WTNetwork
import copy
import numpy as np
import unittest

class TestRandom(unittest.TestCase):
    def test_random_preserving_degree_copies(self):
        weights = np.random.randint(low = -1, high = 2, size = (10,10))
        net = WTNetwork(copy.copy(weights))
        random_preserving_degree(net)
        self.assertTrue(np.array_equal(weights, net.weights))
