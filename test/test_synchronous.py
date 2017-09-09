# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from collections import Counter
from neet.automata import ECA
from neet.boolean import WTNetwork, LogicNetwork
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
from neet.synchronous import *
import numpy as np
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

    def test_trajectory_logicnetwork(self):
        """
        test `trajectory` on `LogicNetwork`s
        """
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), {'01', '10', '11'}),
                            ((0, 1), {'11'})])
        state = [0, 1, 0]
        got = list(trajectory(net, state, 3))
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)
        self.assertEqual([0, 1, 0], state)

        got = list(trajectory(net, state, 3, encode=True))
        self.assertEqual([2, 1, 2, 1], got)

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

    def test_transitions_logicnetwork(self):
        """
        test `transitions` on `LogicNetwork`s
        """
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        got = list(transitions(net))
        self.assertEqual([[1, 0], [1, 1], [1, 0], [1, 1]], got)

    def test_transitions_logicnetwork_encoded(self):
        """
        test `transitions` on `LogicNetwork`s, states encoded
        """
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        got = list(transitions(net, encode=True))
        self.assertEqual([1, 3, 1, 3], got)

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

    def test_attractors_variable_sized(self):
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

    def test_basins_invalid_net(self):
        """
        ``basins`` should raise an error if ``net`` is neither a network nor a
        networkx digraph
        """
        with self.assertRaises(TypeError):
            basins('blah')

        with self.assertRaises(TypeError):
            basins(MockObject)

        with self.assertRaises(TypeError):
            basins(nx.Graph())

    def test_basins_variable_sized(self):
        """
        ``basins`` should raise an error if ``net`` is a variable sized network
        and ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            basins(ECA(30), size=None)

    def test_basins_fixed_sized(self):
        """
        ``basins`` should raise an error if ``net`` is a fized sized network
        and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            basins(MockFixedSizedNetwork, size=5)

    def test_basins_transition_graph(self):
        """
        test ``basins`` on ``s_pombe`` transition graph
        """
        from_graph = basins(transition_graph(s_pombe))
        from_network = basins(s_pombe)

        edges_from_graph = [g.edges() for g in from_graph]
        edges_from_network = [g.edges() for g in from_network]

        self.assertEqual(edges_from_network, edges_from_graph)

    def test_basins_eca(self):
        """
        test ``basins`` on ECAs
        """
        networks = [(ECA(30), 2, [2, 1, 1]), (ECA(30), 3, [8]),
                    (ECA(30), 4, [2, 12, 1, 1]), (ECA(30), 5, [2, 30]),
                    (ECA(30), 6, [62, 1, 1]), (ECA(110), 2, [4]),
                    (ECA(110), 3, [8]), (ECA(110), 4, [4, 6, 6]),
                    (ECA(110), 5, [32]), (ECA(110), 6, [10, 27, 27])]

        for net, width, basin_sizes in networks:
            basin_counter = Counter([len(c) for c in basins(net, size=width)])
            self.assertEqual(Counter(basin_sizes), basin_counter)

    def test_basins_wtnetwork(self):
        """
        test ``basins`` on WTNetworks
        """
        networks = [(s_pombe, [378, 2, 2, 2, 104, 6, 6, 2, 2, 2, 2, 2, 2]),
                    (s_cerevisiae, [7, 1764, 151, 1, 9, 109, 7]),
                    (c_elegans, [4, 219, 12, 5, 16])]

        for net, basin_sizes in networks:
            basin_counter = Counter([len(c) for c in basins(net)])
            self.assertEqual(Counter(basin_sizes), basin_counter)

    def test_basin_entropy_invalid_net(self):
        """
        ``basin_entropy`` should raise an error if ``net`` is neither a network nor a
        networkx digraph
        """
        with self.assertRaises(TypeError):
            basin_entropy('blee')

        with self.assertRaises(TypeError):
            basin_entropy(MockObject)

        with self.assertRaises(TypeError):
            basin_entropy(nx.Graph())

    def test_basin_entropy_variable_sized(self):
        """
        ``basin_entropy`` should raise an error if ``net`` is a variable sized network
        and ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            basin_entropy(ECA(30), size=None)

    def test_basin_entropy_fixed_sized(self):
        """
        ``basin_entropy`` should raise an error if ``net`` is a fized sized network
        and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            basin_entropy(MockFixedSizedNetwork, size=5)

    def test_basin_entropy_transition_graph(self):
        """
        test ``basin_entropy`` on ``s_pombe`` transition graph
        """
        from_graph = basin_entropy(transition_graph(s_pombe))
        from_network = basin_entropy(s_pombe)

        self.assertAlmostEqual(from_network, from_graph)

    def test_basin_entropy_eca(self):
        """
        test ``basin_entropy`` on ECAs
        """
        networks = [(ECA(30), 2, 1.5), (ECA(30), 3, 0.),
                    (ECA(30), 4, 1.186278124459133),
                    (ECA(30), 5, 0.3372900666170139),
                    (ECA(30), 6, 0.23187232431271465),
                    (ECA(110), 2, 0.), (ECA(110), 3, 0.),
                    (ECA(110), 4, 1.561278124459133),
                    (ECA(110), 5, 0.),
                    (ECA(110), 6, 1.4690124052234232)]

        for net, width, entropy in networks:
            self.assertAlmostEqual(basin_entropy(net, size=width), entropy)

    def test_basin_entropy_wtnetwork(self):
        """
        test ``basin_entropy`` on WTNetworks
        """
        networks = [(s_pombe, 1.2218888338849747),
                    (s_cerevisiae, 0.7838577302128783),
                    (c_elegans, 0.8542673572822357),
                    ]

        for net, entropy in networks:
            self.assertAlmostEqual(basin_entropy(net), entropy)

    def test_basin_entropy_wtnetwork_base10(self):
        """
        test ``basin_entropy`` on WTNetworks with different base
        """
        networks = [(s_pombe, 0.36782519036626099),
                    (s_cerevisiae, 0.2359646891271609),
                    (c_elegans, 0.2571600988585521),
                    ]

        for net, entropy in networks:
            self.assertAlmostEqual(basin_entropy(net, base=10), entropy)

    def test_timeseries_not_network(self):
        """
        ``timeseries`` should raise an error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            timeseries(5, timesteps=2)

        with self.assertRaises(TypeError):
            timeseries(MockObject(), timesteps=2)

    def test_timeseries_variable_sized(self):
        """
        ``timeseries`` should raise an error if ``net`` is variable sized and
        ``size`` is ``None``
        """
        with self.assertRaises(ValueError):
            timeseries(ECA(30), size=None, timesteps=5)

    def test_timeseries_fixed_sized(self):
        """
        ``timeseries`` should raise an error if ``net`` is fixed sized and
        ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork, size=5, timesteps=5)

    def test_timeseries_too_short(self):
        """
        ``timeseries`` shoudl raise an error if ``timesteps`` is too small
        """
        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), timesteps=0)

        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), timesteps=-1)

    def test_timeseries_eca(self):
        """
        test ``timeseries`` on ECA
        """
        rule = ECA(30)
        for size in [5, 7, 11]:
            time = 10
            series = timeseries(rule, timesteps=time, size=size)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(rule.state_space(size)):
                for t, expect in enumerate(trajectory(rule, state, timesteps=time)):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_timeseries_wtnetworks(self):
        """
        test ``timeseries`` on WTNetwork
        """
        for (net, size) in [(s_pombe, 9), (s_cerevisiae, 11), (c_elegans, 8)]:
            time = 10
            series = timeseries(net, timesteps=time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(net.state_space()):
                for t, expect in enumerate(trajectory(net, state, timesteps=time)):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))
