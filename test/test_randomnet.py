# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import copy
import networkx as nx
import numpy as np
from neet.boolean.examples import mouse_cortical_7B
from neet.boolean.examples import s_pombe
from neet.boolean.wtnetwork import WTNetwork
from neet.boolean.randomnet import *

TESTSEED = 314159


class TestRandomnet(unittest.TestCase):

    def test_preserving_degrees(self):
        net = s_pombe
        ran_net = rewiring_fixed_degree(net)

        G = nx.from_numpy_matrix(net.weights, create_using = nx.DiGraph())
        ranG = nx.from_numpy_matrix(ran_net.weights, create_using = nx.DiGraph())

        InDegree = list(dict(G.in_degree(weight = 'weight')).values())
        ranInDegree = list(dict(ranG.in_degree(weight = 'weight')).values())

        OutDegree = list(dict(G.out_degree(weight = 'weight')).values())
        ranOutDegree = list(dict(ranG.out_degree(weight = 'weight')).values())

        self.assertEqual(InDegree, ranInDegree)
        self.assertEqual(OutDegree, ranOutDegree)

    def test_preserving_size(self):
        net = s_pombe
        ran_net = rewiring_fixed_size(net)

        EdgeCounts = np.asarray(np.unique(net.weights, return_counts = True))
        ranEdgeCounts = np.asarray(np.unique(ran_net.weights, return_counts = True))

        self.assertTrue(np.array_equal(EdgeCounts, ranEdgeCounts))


    def test_random_logic_invalid_p(self):
        """
        ``random_logic`` should raise a value error if ``p`` is an incorrect size
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
        randnumedges = np.sum([len(randnet.neighbors_in(i)) for i in range(randnet.size)])
        self.assertEqual(numedges, randnumedges)
