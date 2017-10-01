# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from neet.boolean import rannetwork as rnet
from neet.boolean import WTNetwork
import copy
import numpy as np
import networkx as nx
import unittest


class TestRandom(unittest.TestCase):

    def test_copies(self):
        weights = np.random.randint(low = -1, high = 2, size = (10,10))
        net = WTNetwork(copy.copy(weights))

        self.assertTrue(np.array_equal(weights, net.weights))

    def test_preserving_degrees(self):
        weights = np.random.randint(low = -1, high = 2, size = (10,10))
        net = WTNetwork(copy.copy(weights))
        ran_net = rnet.rewiring_fixed_degree(net)

        G = nx.from_numpy_matrix(net.weights, create_using = nx.DiGraph())
        ranG = nx.from_numpy_matrix(ran_net.weights, create_using = nx.DiGraph())

        InDegree = list(G.in_degree(weight = 'weight').values())
        ranInDegree = list(ranG.in_degree(weight = 'weight').values())

        OutDegree = list(G.out_degree(weight = 'weight').values())
        ranOutDegree = list(ranG.out_degree(weight = 'weight').values())

        self.assertEqual(InDegree, ranInDegree)
        self.assertEqual(OutDegree, ranOutDegree)

    def test_preserving_size(self):
        weights = np.random.randint(low = -1, high = 2, size = (10,10))
        net = WTNetwork(copy.copy(weights))
        ran_net = rnet.rewiring_fixed_size(net)

        EdgeCounts = np.asarray(np.unique(net.weights, return_counts = True))
        ranEdgeCounts = np.asarray(np.unique(ran_net.weights, return_counts = True))

        self.assertTrue(np.array_equal(EdgeCounts, ranEdgeCounts))
