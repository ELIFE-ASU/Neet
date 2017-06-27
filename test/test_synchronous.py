# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import unittest
from collections import Counter
from neet.automata import ECA
from neet.boolean import WTNetwork
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
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
        """
        ``transitions_graph`` should raise an error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            transition_graph(MockObject())

    def test_transition_graph_variable_sized(self):
        """
        ``transitions_graph`` should raise an error if ``net`` is variable sized
        and ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            transition_graph(ECA(30))

    def test_transition_graph_fixed_sized(self):
        """
        ``transitions_graph`` should raise an error if ``net`` is fixed sized
        and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            transition_graph(MockFixedSizedNetwork(), size=5)

    def test_transition_graph_eca(self):
        """
        test ``transitions_graph`` on ``ECA``
        """
        graph = transition_graph(ECA(30), size=8)
        self.assertEqual(256, graph.number_of_nodes())
        self.assertEqual(256, graph.number_of_edges())

    def test_transition_graph_s_pombe(self):
        """
        test ``transitions_graph`` on ``s_pombe``
        """
        volume = s_pombe.state_space().volume
        graph = transition_graph(s_pombe)
        self.assertEqual(volume, graph.number_of_nodes())
        self.assertEqual(volume, graph.number_of_edges())

    def test_attractors_invalid_net(self):
        """
        ``attractors`` should raise an error if ``net`` is neither a network
        nor a networkx digraph
        """
        with self.assertRaises(TypeError):
            attractors('blah')

        with self.assertRaises(TypeError):
            attractors(MockObject())

        with self.assertRaises(TypeError):
            attractors(nx.Graph())

    def test_attractors_varialbe_sized(self):
        """
        ``attractors`` should raise an error if ``net`` is a variable sized
        network and ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            attractors(ECA(30), size=None)

    def test_attractors_fixed_sized(self):
        """
        ``attractors`` should raise an error if ``net`` is either a fixed sized
        network or a networkx digraph, and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            attractors(MockFixedSizedNetwork(), size=5)

        with self.assertRaises(ValueError):
            attractors(nx.DiGraph(), size=5)

    def test_attractors_eca(self):
        """
        test ``attractors`` on ECA
        """
        networks = [(ECA(30), 2, 3), (ECA(30), 3, 1), (ECA(30), 4, 4),
                    (ECA(30), 5, 2), (ECA(30), 6, 3), (ECA(110), 2, 1),
                    (ECA(110), 3, 1), (ECA(110), 4, 3), (ECA(110), 5, 1),
                    (ECA(110), 6, 3)]
        for rule, width, size in networks:
            self.assertEqual(size, len(list(attractors(rule, width))))

    def test_attractors_wtnetworks(self):
        """
        test ``attractors`` on WTNetworks
        """
        networks = [(s_pombe, 13), (s_cerevisiae, 7), (c_elegans, 5)]
        for net, size in networks:
            self.assertEqual(size, len(list(attractors(net))))

    def test_attractors_transition_graph(self):
        """
        test ``attractors`` on ``s_pombe`` transition graph
        """
        att_from_graph = list(attractors(transition_graph(s_pombe)))
        att_from_network = list(attractors(s_pombe))
        self.assertEqual(att_from_network, att_from_graph)
    
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
