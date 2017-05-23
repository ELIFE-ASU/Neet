# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.landscape import *
import numpy as np
from neet.boolean.examples import s_pombe
from collections import Counter

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self, lattice):
            pass
        def state_space(self):
            return StateSpace(1)

    class IsNotNetwork(object):
        pass

    def test_trajectory_not_network(self):
        with self.assertRaises(TypeError):
            list(trajectory(5, [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(self.IsNotNetwork(), [1,2,3]))

        with self.assertRaises(TypeError):
            list(trajectory(self.IsNetwork, [1,2,3]))

    def test_trajectory_too_short(self):
        with self.assertRaises(ValueError):
            list(trajectory(self.IsNetwork(), [1,2,3], n=0))

        with self.assertRaises(ValueError):
            list(trajectory(self.IsNetwork(), [1,2,3], n=-1))

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
            list(transitions(self.IsNotNetwork(), StateSpace(5)))

    def test_transitions_not_statespace(self):
        with self.assertRaises(TypeError):
            list(transitions(self.IsNetwork(), 5))

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
            transition_graph(self.IsNotNetwork())

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
                            
    def test_basins(self):
        
        b = basins(s_pombe)
        
        s_pombe_counter = Counter([378, 2, 2, 2, 104,6, 6,
                                   2, 2, 2, 2, 2, 2])
        b_counter = Counter([ len(c) for c in b ])
                                  
        self.assertEqual(s_pombe_counter,b_counter)



