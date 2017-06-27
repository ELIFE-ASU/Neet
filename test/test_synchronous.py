# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest
from collections import Counter
from neet.automata import ECA
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
from neet.synchronous import *
from .mock import MockObject, MockFixedSizedNetwork

class TestCore(unittest.TestCase):
    def test_trajectory_not_network(self):
        with self.assertRaises(TypeError):
            list(trajectory(5, [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(MockObject(), [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(MockFixedSizedNetwork, [1,2,3]))

    def test_trajectory_too_short(self):
        with self.assertRaises(ValueError):
            list(trajectory(MockFixedSizedNetwork(), [1,2,3], n=0))

        with self.assertRaises(ValueError):
            list(trajectory(MockFixedSizedNetwork(), [1,2,3], n=-1))

    def test_trajectory_eca_not_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, []))

        xs = [0,1,0]
        got = list(trajectory(rule30, xs))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([[0,1,0],[1,1,1]], got)

        got = list(trajectory(rule30, xs, n=2))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([[0,1,0],[1,1,1],[0,0,0]], got)

    def test_trajectory_eca_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, [], encode=True))

        xs = [0,1,0]
        got = list(trajectory(rule30, xs, encode=True))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([2,7], got)

        got = list(trajectory(rule30, xs, n=2, encode=True))
        self.assertEqual([0,1,0], xs)
        self.assertEqual([2,7,0], got)

    def test_trajectory_wtnetwork_not_encoded(self):
        from neet.boolean import WTNetwork
        net = WTNetwork(
            weights    = [[1,0],[-1,0]],
            thresholds = [0.5,0.0],
            theta      = WTNetwork.positive_threshold
        )

        xs = [0,0]
        got = list(trajectory(net, xs))
        self.assertEqual([0,0], xs)
        self.assertEqual([[0,0],[0,1]], got)

        got = list(trajectory(net, xs, n=3))
        self.assertEqual([0,0], xs)
        self.assertEqual([[0,0],[0,1],[0,1],[0,1]], got)

    def test_trajectory_wtnetwork_encoded(self):
        from neet.boolean import WTNetwork
        net = WTNetwork(
            weights    = [[1,0],[-1,0]],
            thresholds = [0.5,0.0],
            theta      = WTNetwork.positive_threshold
        )

        xs = [0,0]
        got = list(trajectory(net, xs, encode=True))
        self.assertEqual([0,0], xs)
        self.assertEqual([0,2], got)

        got = list(trajectory(net, xs, n=3, encode=True))
        self.assertEqual([0,0], xs)
        self.assertEqual([0,2,2,2], got)

    def test_transitions_not_network(self):
        with self.assertRaises(TypeError):
            list(transitions(MockObject(), StateSpace(5)))

    def test_transitions_not_statespace(self):
        with self.assertRaises(TypeError):
            list(transitions(MockFixedSizedNetwork(), 5))

    def test_transitions_eca_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)

        got = list(transitions(rule30, n=1))
        self.assertEqual([0,0], got)

        got = list(transitions(rule30, n=2))
        self.assertEqual([0,1,2,0], got)

        got = list(transitions(rule30, n=3))
        self.assertEqual([0,7,7,1,7,4,2,0], got)

    def test_transitions_eca_not_encoded(self):
        from neet.automata import ECA
        rule30 = ECA(30)

        got = list(transitions(rule30, n=1, encode=False))
        self.assertEqual([[0],[0]], got)

        got = list(transitions(rule30, n=2, encode=False))
        self.assertEqual([[0,0],[1,0],[0,1],[0,0]], got)

        got = list(transitions(rule30, n=3, encode=False))
        self.assertEqual([[0,0,0],[1,1,1],[1,1,1],[1,0,0]
                         ,[1,1,1],[0,0,1],[0,1,0],[0,0,0]], got)

    def test_transitions_wtnetwork_encoded(self):
        from neet.boolean import WTNetwork
        net = WTNetwork(
            weights    = [[1,0],[-1,1]],
            thresholds = [0.5,0.0],
            theta      = WTNetwork.positive_threshold
        )

        with self.assertRaises(TypeError):
            list(transitions(net, n=1))

        got = list(transitions(net))
        self.assertEqual([2,1,2,3], got)

    def test_transitions_eca_not_encoded(self):
        from neet.boolean import WTNetwork
        net = WTNetwork(
            weights    = [[1,0],[-1,1]],
            thresholds = [0.5,0.0],
            theta      = WTNetwork.positive_threshold
        )

        with self.assertRaises(TypeError):
            list(transitions(net, n=1, encode=False))

        got = list(transitions(net, encode=False))
        self.assertEqual([[0,1],[1,0],[0,1],[1,1]], got)

    def test_transition_graph_not_network(self):
        with self.assertRaises(TypeError):
            transition_graph(MockObject())

    def test_transition_graph_s_pombe(self):
        g = transition_graph(s_pombe)

        # the transition graph should have number of nodes
        # equal to the volume of state space (the number of
        # possible states)
        self.assertEqual(s_pombe.state_space().volume,
                         g.number_of_nodes())

    def test_attractors_s_pombe(self):
        att = list( attractors(s_pombe) )

        self.assertEqual(13, len(att))
    
    def test_attractors_type(self):
        att_from_graph = attractors(transition_graph(s_pombe))
        att_from_network = attractors(s_pombe)
        self.assertEqual(list(att_from_network),list(att_from_graph))
    
    def test_attractors_typeerror(self):
        with self.assertRaises(TypeError):
            attractors('blah')
        
        with self.assertRaises(TypeError):
            attractors(nx.Graph()) # (undirected)
    
    def test_basins(self):
        
        b = basins(s_pombe)
        
        s_pombe_counter = Counter([378, 2, 2, 2, 104,6, 6,
                                   2, 2, 2, 2, 2, 2])
        b_counter = Counter([ len(c) for c in b ])
                                  
        self.assertEqual(s_pombe_counter,b_counter)

    def test_basins_type(self):
        b_from_graph = basins(transition_graph(s_pombe))
        b_from_network = basins(s_pombe)
        
        edges_from_graph = [ g.edges() for g in b_from_graph ]
        edges_from_network = [ g.edges() for g in b_from_network ]
        
        self.assertEqual(edges_from_network,edges_from_graph)

    def test_basins_typeerror(self):
        with self.assertRaises(TypeError):
            basins('blah')

        with self.assertRaises(TypeError):
            basins(nx.Graph()) # (undirected)


    def test_timeseries_not_network(self):
        with self.assertRaises(TypeError):
            timeseries(5, timesteps=2)

        with self.assertRaises(TypeError):
            timeseries(MockObject(), timesteps=2)


    def test_timeseries_not_fixed_sized(self):
        with self.assertRaises(ValueError):
            timeseries(ECA(30), timesteps=5)


    def test_timeseries_fixed_sized_with_size(self):
        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), size=5, timesteps=5)


    def test_timeseries_too_short(self):
        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), timesteps=0)

        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), timesteps=-1)


    def test_timeseries_wtnetworks(self):
        for (net, size) in [(s_pombe,9), (s_cerevisiae,11), (c_elegans,8)]:
            time = 10
            series = timeseries(net, timesteps=time)
            self.assertEqual((size, 2**size, time+1), series.shape)
            for (index, state) in enumerate(net.state_space().states()):
                for (t, expect) in enumerate(trajectory(net, state, n=time)):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))


    def test_timeseries_eca(self):
        rule = ECA(30)
        for size in [5,7,11]:
            time = 10
            series = timeseries(rule, timesteps=time, size=size)
            self.assertEqual((size, 2**size, time+1), series.shape)
            for (index, state) in enumerate(rule.state_space(size).states()):
                for t, expect in enumerate(trajectory(rule, state, n=time)):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))
