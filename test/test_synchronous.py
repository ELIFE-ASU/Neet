# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest
from collections import Counter
from neet.automata import ECA
from neet.boolean import WTNetwork
from neet.boolean.examples import s_pombe
from neet.synchronous import *
from .mock import MockObject, MockFixedSizedNetwork

class TestSynchronous(unittest.TestCase):
    """
    Unit tests for the ``neet.synchronous`` module
    """
    def test_trajectory_not_network(self):
        """
        ``trajectory`` should raise a type error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            list(trajectory(5, [1, 2, 3]))

        with self.assertRaises(TypeError):
            list(trajectory(MockObject(), [1, 2, 3]))

        with self.assertRaises(TypeError):
            list(trajectory(MockFixedSizedNetwork, [1, 2, 3]))

    def test_trajectory_too_short(self):
        """
        ``trajectory`` should raise a value error if ``timeseries`` is less
        than 1
        """
        with self.assertRaises(ValueError):
            list(trajectory(MockFixedSizedNetwork(), [1, 2, 3], timesteps=0))

        with self.assertRaises(ValueError):
            list(trajectory(MockFixedSizedNetwork(), [1, 2, 3], timesteps=-1))

    def test_trajectory_eca(self):
        """
        test ``trajectory`` on ECAs
        """
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, []))

        xs = [0, 1, 0]
        got = list(trajectory(rule30, xs))
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1]], got)

        got = list(trajectory(rule30, xs, timesteps=2))
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

    def test_trajectory_eca_encoded(self):
        """
        test ``trajectory`` on ECAs; encoding the states
        """
        rule30 = ECA(30)
        with self.assertRaises(ValueError):
            list(trajectory(rule30, [], encode=True))

        state = [0, 1, 0]
        got = list(trajectory(rule30, state, encode=True))
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 7], got)

        got = list(trajectory(rule30, state, timesteps=2, encode=True))
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 7, 0], got)

    def test_trajectory_wtnetwork(self):
        """
        test ``trajectory`` on WTNetworks
        """
        net = WTNetwork(
            weights=[[1, 0], [-1, 0]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        state = [0, 0]
        got = list(trajectory(net, state))
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = list(trajectory(net, state, timesteps=3))
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

    def test_trajectory_wtnetwork_encoded(self):
        """
        test ``trajectory`` on WTNetworks; encoding the states
        """
        net = WTNetwork(
            weights=[[1, 0], [-1, 0]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        state = [0, 0]
        got = list(trajectory(net, state, encode=True))
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2], got)

        got = list(trajectory(net, state, timesteps=3, encode=True))
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2, 2, 2], got)

    def test_transitions_not_network(self):
        """
        ``transitions`` should raise a type error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            list(transitions(MockObject(), 5))

    def test_transitions_not_fixed_sized(self):
        """
        ``transitions`` should raise an error if ``net`` is not fixed sized
        and ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            list(transitions(ECA(30), size=None))

    def test_transitions_fixed_sized(self):
        """
        ``transitions`` should raise an error if ``net`` is fixed sized and
        ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            list(transitions(MockFixedSizedNetwork, size=3))

    def test_transitions_eca(self):
        """
        test ``transitions`` on ECAs; encoding the states
        """
        rule30 = ECA(30)

        got = list(transitions(rule30, size=1))
        self.assertEqual([[0], [0]], got)

        got = list(transitions(rule30, size=2))
        self.assertEqual([[0, 0], [1, 0], [0, 1], [0, 0]], got)

        got = list(transitions(rule30, size=3))
        self.assertEqual([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0],
                          [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]], got)

    def test_transitions_eca_encoded(self):
        """
        test ``transitions`` on ECAs; encoding the states
        """
        rule30 = ECA(30)

        got = list(transitions(rule30, size=1, encode=True))
        self.assertEqual([0, 0], got)

        got = list(transitions(rule30, size=2, encode=True))
        self.assertEqual([0, 1, 2, 0], got)

        got = list(transitions(rule30, size=3, encode=True))
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], got)

    def test_transitions_wtnetwork(self):
        """
        test ``transitions`` on WTNetworks
        """
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        got = list(transitions(net))
        self.assertEqual([[0, 1], [1, 0], [0, 1], [1, 1]], got)

    def test_transitions_wtnetwork_encoded(self):
        """
        test ``transitions`` on WTNetworks; encoding the states
        """
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        got = list(transitions(net, encode=True))
        self.assertEqual([2, 1, 2, 3], got)

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
