# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.boolean.examples import mouse_cortical_7B
from neet.boolean.randomnet import *

import numpy as np

TESTSEED = 314159

class TestRandomnet(unittest.TestCase):
    

    def test_random_logic_invalid_p(self):
        """
        ``random_logic`` should raise a value error if ``p`` is an incorrect size
        """
        with self.assertRaises(ValueError):
            net = mouse_cortical_7B
            random_logic(net,p=np.ones(net.size+1))

    def test_random_logic_fixed_structure(self):
        net = mouse_cortical_7B
        np.random.seed(TESTSEED)
        randnet = random_logic(net,connections='fixed-structure')
        # fixed-structure should preserve all neighbors
        for i in range(net.size):
            self.assertEqual(net.neighbors_in(i),randnet.neighbors_in(i))

    def test_random_logic_fixed_in_degree(self):
        net = mouse_cortical_7B
        np.random.seed(TESTSEED)
        randnet = random_logic(net,connections='fixed-in-degree')
        # fixed-in-degree should preserve each node's in degree
        for i in range(net.size):
            self.assertEqual(len(net.neighbors_in(i)),
                             len(randnet.neighbors_in(i)))

