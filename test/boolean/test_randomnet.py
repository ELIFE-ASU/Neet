import unittest
import numpy as np
from neet.boolean.examples import mouse_cortical_7B
from neet.boolean.randomnet import random_logic, random_binary_states

TESTSEED = 314159


class TestRandomnet(unittest.TestCase):
    def test_random_logic_invalid_p(self):
        """
        ``random_logic`` should raise a value error if ``p`` is an
        incorrect size
        """
        with self.assertRaises(ValueError):
            net = mouse_cortical_7B
            random_logic(net, p=np.ones(net.size + 1))

    def test_random_binary_states(self):
        self.assertEqual(8, len(random_binary_states(4, 0.5)))
        self.assertTrue(len(random_binary_states(3, 0.4)) in (3, 4))

    def test_random_logic_fixed_structure(self):
        net = mouse_cortical_7B
        np.random.seed(TESTSEED)
        randnet = random_logic(net, connections='fixed-structure')
        # fixed-structure should preserve all neighbors
        for i in range(net.size):
            self.assertEqual(net.neighbors_in(i), randnet.neighbors_in(i))

    def test_random_logic_fixed_in_degree(self):
        net = mouse_cortical_7B
        np.random.seed(TESTSEED)
        randnet = random_logic(net, connections='fixed-in-degree')
        # fixed-in-degree should preserve each node's in degree
        for i in range(net.size):
            self.assertEqual(len(net.neighbors_in(i)),
                             len(randnet.neighbors_in(i)))

    def test_random_logic_fixed_mean_degree(self):
        net = mouse_cortical_7B
        np.random.seed(TESTSEED)
        randnet = random_logic(net, connections='fixed-mean-degree')
        # fixed-mean-degree should preserve the total number of edges
        numedges = np.sum([len(net.neighbors_in(i)) for i in range(net.size)])
        randnumedges = np.sum([len(randnet.neighbors_in(i))
                               for i in range(randnet.size)])
        self.assertEqual(numedges, randnumedges)
