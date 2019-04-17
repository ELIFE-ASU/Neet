import unittest
from collections import Counter
from neet.automata import ECA
from neet.boolean import WTNetwork, LogicNetwork
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
from neet.synchronous import (trajectory, transitions, transition_graph,
                              attractors, basins, basin_entropy,
                              timeseries, Landscape)
from neet.statespace import StateSpace
import numpy as np
import networkx as nx
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
            trajectory(5, [1, 2, 3])

        with self.assertRaises(TypeError):
            trajectory(MockObject(), [1, 2, 3])

        with self.assertRaises(TypeError):
            trajectory(MockFixedSizedNetwork, [1, 2, 3])

    def test_trajectory_too_short(self):
        """
        ``trajectory`` should raise a value error if ``timeseries`` is less
        than 1
        """
        with self.assertRaises(ValueError):
            trajectory(MockFixedSizedNetwork(), [1, 2, 3], timesteps=0)

        with self.assertRaises(ValueError):
            trajectory(MockFixedSizedNetwork(), [1, 2, 3], timesteps=-1)

    def test_trajectory_eca(self):
        """
        test ``trajectory`` on ECAs
        """
        rule30 = ECA(30, 3)
        with self.assertRaises(ValueError):
            trajectory(rule30, [])

        xs = [0, 1, 0]
        got = trajectory(rule30, xs)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1]], got)

        got = trajectory(rule30, xs, timesteps=2)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

    def test_trajectory_eca_encoded(self):
        """
        test ``trajectory`` on ECAs; encoding the states
        """
        rule30 = ECA(30, 3)
        with self.assertRaises(ValueError):
            trajectory(rule30, [], encode=True)

        state = [0, 1, 0]
        got = trajectory(rule30, state, encode=True)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 7], got)

        got = trajectory(rule30, state, timesteps=2, encode=True)
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
        got = trajectory(net, state)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = trajectory(net, state, timesteps=3)
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
        got = trajectory(net, state, encode=True)
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2], got)

        got = trajectory(net, state, timesteps=3, encode=True)
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
        got = trajectory(net, state, 3)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)
        self.assertEqual([0, 1, 0], state)

        got = list(trajectory(net, state, 3, encode=True))
        self.assertEqual([2, 1, 2, 1], got)

    def test_transitions_not_network(self):
        """
        ``transitions`` should raise a type error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            transitions(MockObject(), 5)

    def test_transitions_fixed_sized(self):
        """
        ``transitions`` should raise an error if ``net`` is fixed sized and
        ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            transitions(MockFixedSizedNetwork(), size=3)

    def test_transitions_eca(self):
        """
        test ``transitions`` on ECAs; encoding the states
        """
        rule30 = ECA(30, 1)

        got = transitions(rule30)
        self.assertEqual([[0], [0]], got)

        rule30.size = 2
        got = transitions(rule30)
        self.assertEqual([[0, 0], [1, 0], [0, 1], [0, 0]], got)

        rule30.size = 3
        got = transitions(rule30)
        self.assertEqual([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0],
                          [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]], got)

    def test_transitions_eca_encoded(self):
        """
        test ``transitions`` on ECAs; encoding the states
        """
        got = transitions(ECA(30, 1), encode=True)
        self.assertEqual([0, 0], got)

        got = transitions(ECA(30, 2), encode=True)
        self.assertEqual([0, 1, 2, 0], got)

        got = transitions(ECA(30, 3), encode=True)
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

        got = transitions(net)
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

        got = transitions(net, encode=True)
        self.assertEqual([2, 1, 2, 3], got)

    def test_transitions_logicnetwork(self):
        """
        test `transitions` on `LogicNetwork`s
        """
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        got = transitions(net)
        self.assertEqual([[1, 0], [1, 1], [1, 0], [1, 1]], got)

    def test_transitions_logicnetwork_encoded(self):
        """
        test `transitions` on `LogicNetwork`s, states encoded
        """
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
        got = transitions(net, encode=True)
        self.assertEqual([1, 3, 1, 3], got)

    def test_transition_graph_not_network(self):
        """
        ``transitions_graph`` should raise an error if ``net`` is not a network
        """
        with self.assertRaises(TypeError):
            transition_graph(MockObject())

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
        graph = transition_graph(ECA(30, 8))
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

    def test_attractors_fixed_sized(self):
        """
        ``attractors`` should raise an error if ``net`` is either a fixed sized
        network or a networkx digraph, and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            attractors(MockFixedSizedNetwork(), size=5)

    def test_attractors_eca(self):
        """
        test ``attractors`` on ECA
        """
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]
        for rule, size in networks:
            self.assertEqual(size, len(attractors(rule)))

    def test_attractors_wtnetworks(self):
        """
        test ``attractors`` on WTNetworks
        """
        networks = [(s_pombe, 13), (s_cerevisiae, 7), (c_elegans, 5)]
        for net, size in networks:
            self.assertEqual(size, len(attractors(net)))

    def test_attractors_transition_graph(self):
        """
        test ``attractors`` on ``s_pombe`` transition graph
        """
        att_from_graph = attractors(transition_graph(s_pombe))
        att_from_network = attractors(s_pombe)

        for (a, b) in zip(att_from_graph, att_from_network):
            a.sort()
            b.sort()

        att_from_graph.sort()
        att_from_network.sort()

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

    def test_basins_fixed_sized(self):
        """
        ``basins`` should raise an error if ``net`` is a fized sized network
        and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            basins(MockFixedSizedNetwork(), size=5)

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
        networks = [(ECA(30, 2), [2, 1, 1]), (ECA(30, 3), [8]),
                    (ECA(30, 4), [2, 12, 1, 1]), (ECA(30, 5), [2, 30]),
                    (ECA(30, 6), [62, 1, 1]), (ECA(110, 2), [4]),
                    (ECA(110, 3), [8]), (ECA(110, 4), [4, 6, 6]),
                    (ECA(110, 5), [32]), (ECA(110, 6), [10, 27, 27])]

        for net, basin_sizes in networks:
            basin_counter = Counter([len(c) for c in basins(net)])
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
        ``basin_entropy`` should raise an error if ``net`` is neither
        a network nor a networkx digraph
        """
        with self.assertRaises(TypeError):
            basin_entropy('blee')

        with self.assertRaises(TypeError):
            basin_entropy(MockObject)

        with self.assertRaises(TypeError):
            basin_entropy(nx.Graph())

    def test_basin_entropy_fixed_sized(self):
        """
        ``basin_entropy`` should raise an error if ``net`` is a fized
        sized network and ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            basin_entropy(MockFixedSizedNetwork(), size=5)

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
        networks = [(ECA(30, 2), 1.5), (ECA(30, 3), 0.),
                    (ECA(30, 4), 1.186278124459133),
                    (ECA(30, 5), 0.3372900666170139),
                    (ECA(30, 6), 0.23187232431271465),
                    (ECA(110, 2), 0.), (ECA(110, 3), 0.),
                    (ECA(110, 4), 1.561278124459133),
                    (ECA(110, 5), 0.),
                    (ECA(110, 6), 1.4690124052234232)]

        for net, entropy in networks:
            self.assertAlmostEqual(basin_entropy(net), entropy)

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

    def test_timeseries_fixed_sized(self):
        """
        ``timeseries`` should raise an error if ``net`` is fixed sized and
        ``size`` is not ``None``
        """
        with self.assertRaises(ValueError):
            timeseries(MockFixedSizedNetwork(), size=5, timesteps=5)

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
        for size in [5, 7, 11]:
            rule = ECA(30, size)
            time = 10
            series = timeseries(rule, timesteps=time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(rule.state_space()):
                traj = trajectory(rule, state, timesteps=time)
                for t, expect in enumerate(traj):
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
                traj = trajectory(net, state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))


class TestLandscape(unittest.TestCase):
    """
    Test the ``neet.synchronous.Landscape`` class
    """

    def test_canary(self):
        """
        Canary test
        """
        self.assertTrue(True)

    def test_init_not_network(self):
        """
        ``Landscape.__init__`` should raise a type error if ``net`` is not a
        network.
        """
        with self.assertRaises(TypeError):
            Landscape(MockObject())

        with self.assertRaises(TypeError):
            Landscape(MockObject(), size=5)

    def test_init_fixed_sized(self):
        """
        ``Landscape.__init__`` should raise a value errorif ``net`` is fixed
        sized, but ``size`` is not ``None``.
        """
        with self.assertRaises(ValueError):
            Landscape(MockFixedSizedNetwork(), size=3)

    def test_transitions_eca(self):
        ca = ECA(30, 1)

        landscape = Landscape(ca)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(1, landscape.size)
        self.assertEqual([0, 0], list(landscape.transitions))

        ca.size = 2
        landscape = Landscape(ca)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 0], list(landscape.transitions))

        ca.size = 3
        landscape = Landscape(ca)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

        ca.size = 10
        landscape = Landscape(ca)
        trans = landscape.transitions
        self.assertEqual(ca, landscape.network)
        self.assertEqual(10, landscape.size)
        self.assertEqual(1024, len(trans))
        self.assertEqual([0, 515, 7, 517, 14, 525, 11,
                          521, 28, 543], list(trans[:10]))
        self.assertEqual([18, 16, 13, 14, 10, 8, 7, 4, 2, 0],
                         list(trans[-10:]))

    def test_transitions_eca_index(self):
        ca = ECA(30, 3)
        landscape = Landscape(ca, index=1)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 3, 2, 1, 6, 5, 6, 5], list(landscape.transitions))

        landscape = Landscape(ca, index=0)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 1, 3, 3, 5, 4, 6, 6], list(landscape.transitions))

        landscape = Landscape(ca, index=None)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_eca_pin(self):
        ca = ECA(30, 3)
        landscape = Landscape(ca, pin=[1])
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 5, 7, 3, 5, 4, 2, 2], list(landscape.transitions))

        landscape = Landscape(ca, pin=[0])
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 6, 1, 6, 5, 2, 1], list(landscape.transitions))

        landscape = Landscape(ca, pin=None)
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_eca_values(self):
        ca = ECA(30, 3)
        landscape = Landscape(ca, values={0: 1})
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([1, 7, 7, 1, 7, 5, 3, 1], list(landscape.transitions))

        landscape = Landscape(ca, values={1: 0})
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 5, 5, 1, 5, 4, 0, 0], list(landscape.transitions))

        landscape = Landscape(ca, values={})
        self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_wtnetwork(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = Landscape(net)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_index(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = Landscape(net, index=1)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        landscape = Landscape(net, index=0)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 3], list(landscape.transitions))

        landscape = Landscape(net, index=None)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_pin(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = Landscape(net, pin=[1])
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 3], list(landscape.transitions))

        landscape = Landscape(net, pin=[0])
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        landscape = Landscape(net, pin=None)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_values(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = Landscape(net, values={0: 1})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([3, 1, 3, 3], list(landscape.transitions))

        landscape = Landscape(net, values={1: 0})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 0, 1], list(landscape.transitions))

        landscape = Landscape(net, values={})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        # Add more test for different thetas?

    def test_transitions_logicnetwork(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = Landscape(net)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_index(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = Landscape(net, index=1)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 3, 0, 3], list(landscape.transitions))

        landscape = Landscape(net, index=0)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 3, 3], list(landscape.transitions))

        landscape = Landscape(net, index=None)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_pin(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = Landscape(net, pin=[1])
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 3, 3], list(landscape.transitions))

        landscape = Landscape(net, pin=[0])
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 3, 0, 3], list(landscape.transitions))

        landscape = Landscape(net, pin=None)
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_values(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = Landscape(net, values={0: 1})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

        landscape = Landscape(net, values={1: 0})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 1, 1], list(landscape.transitions))

        landscape = Landscape(net, values={})
        self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

        # Add more test for different thetas?

    def test_transitions_spombe(self):
        landscape = Landscape(s_pombe)
        self.assertEqual(s_pombe, landscape.network)
        self.assertEqual(9, landscape.size)
        trans = landscape.transitions
        self.assertEqual(512, len(trans))
        self.assertEqual([2, 2, 130, 130, 4, 0, 128,
                          128, 8, 0], list(trans[:10]))
        self.assertEqual([464, 464, 344, 336, 464, 464, 348,
                          336, 464, 464], list(trans[-10:]))

    def test_is_state_space(self):
        self.assertTrue(issubclass(Landscape, StateSpace))

        space = s_pombe.state_space()
        landscape = Landscape(s_pombe)
        self.assertEqual(list(space), list(landscape))
        self.assertEqual([space.encode(state) for state in space],
                         list(map(landscape.encode, landscape)))

        ca = ECA(30, 10)
        space = ca.state_space()
        landscape = Landscape(ca)
        self.assertEqual(list(space), list(landscape))
        self.assertEqual([space.encode(state) for state in space],
                         list(map(landscape.encode, landscape)))

    def test_attractors_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            landscape = Landscape(rule)
            self.assertEqual(size, len(landscape.attractors))

    def test_basins_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            landscape = Landscape(rule)
            self.assertEqual(landscape.volume, len(landscape.basins))
            self.assertEqual(size, 1 + np.max(landscape.basins))

            unique = list(set(landscape.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_attractors_wtnetworks(self):
        networks = [(s_pombe, 13), (s_cerevisiae, 7), (c_elegans, 5)]
        for net, size in networks:
            landscape = Landscape(net)
            self.assertEqual(size, len(landscape.attractors))
            self.assertEqual(landscape.volume, len(landscape.basins))
            self.assertEqual(size, 1 + np.max(landscape.basins))

            unique = list(set(landscape.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_trajectory_too_short(self):
        landscape = Landscape(ECA(30, 3))
        with self.assertRaises(ValueError):
            landscape.trajectory([0, 0, 0], timesteps=-1)

        with self.assertRaises(ValueError):
            landscape.trajectory([0, 0, 0], timesteps=0)

    def test_trajectory_eca(self):
        landscape = Landscape(ECA(30, 3))

        with self.assertRaises(ValueError):
            landscape.trajectory([])

        with self.assertRaises(ValueError):
            landscape.trajectory([0, 1])

        xs = [0, 1, 0]

        got = landscape.trajectory(xs)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = landscape.trajectory(xs, timesteps=5)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

        got = landscape.trajectory(xs, encode=False)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = landscape.trajectory(xs, timesteps=5, encode=False)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

        got = landscape.trajectory(xs, encode=True)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([2, 7, 0], list(got))

        got = landscape.trajectory(xs, timesteps=5, encode=True)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        xs = 2
        got = landscape.trajectory(xs)
        self.assertEqual([2, 7, 0], list(got))

        got = landscape.trajectory(xs, timesteps=5)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        got = landscape.trajectory(xs, encode=True)
        self.assertEqual([2, 7, 0], list(got))

        got = landscape.trajectory(xs, timesteps=5, encode=True)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        got = landscape.trajectory(xs, encode=False)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = landscape.trajectory(xs, timesteps=5, encode=False)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

    def test_trajectory_wtnetwork(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 0]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = Landscape(net)

        state = [0, 0]
        got = landscape.trajectory(state)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = landscape.trajectory(state, timesteps=3)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

        got = landscape.trajectory(state, encode=False)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = landscape.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

        got = landscape.trajectory(state, encode=True)
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2], got)

        got = landscape.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2, 2, 2], got)

        state = 0
        got = landscape.trajectory(state)
        self.assertEqual([0, 2], got)

        got = landscape.trajectory(state, timesteps=3)
        self.assertEqual([0, 2, 2, 2], got)

        got = landscape.trajectory(state, encode=True)
        self.assertEqual([0, 2], got)

        got = landscape.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 2, 2, 2], got)

        got = landscape.trajectory(state, encode=False)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = landscape.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

    def test_trajectory_logicnetwork(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), {'01', '10', '11'}),
                            ((0, 1), {'11'})])

        landscape = Landscape(net)

        state = [0, 1, 0]
        got = landscape.trajectory(state, encode=False)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = landscape.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

        got = landscape.trajectory(state)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = landscape.trajectory(state, timesteps=3)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

        got = landscape.trajectory(state, encode=True)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 1], got)

        got = landscape.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 1, 2, 1], got)

        state = 2
        got = landscape.trajectory(state)
        self.assertEqual([2, 1], got)

        got = landscape.trajectory(state, timesteps=3)
        self.assertEqual([2, 1, 2, 1], got)

        got = landscape.trajectory(state, encode=True)
        self.assertEqual([2, 1], got)

        got = landscape.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([2, 1, 2, 1], got)

        got = landscape.trajectory(state, encode=False)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = landscape.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

    def test_timeseries_too_short(self):
        landscape = Landscape(ECA(30, 3))
        with self.assertRaises(ValueError):
            landscape.timeseries(-1)

        with self.assertRaises(ValueError):
            landscape.timeseries(0)

    def test_timeseries_eca(self):
        for size in [5, 7, 11]:
            rule = ECA(30, size)
            landscape = Landscape(rule)
            time = 10
            series = landscape.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(rule.state_space()):
                traj = landscape.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_timeseries_wtnetworks(self):
        for net, size in [(s_pombe, 9), (s_cerevisiae, 11), (c_elegans, 8)]:
            landscape = Landscape(net)
            time = 10
            series = landscape.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(net.state_space()):
                traj = landscape.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_basin_sizes(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            basins = landscape.basins
            histogram = [0] * (np.max(basins) + 1)
            for b in basins:
                histogram[b] += 1
            self.assertEqual(histogram, list(landscape.basin_sizes))

    def test_basin_entropy_eca(self):
        networks = [(ECA(30, 2), 1.500000, 0.451545),
                    (ECA(30, 3), 0.000000, 0.000000),
                    (ECA(30, 4), 1.186278, 0.357105),
                    (ECA(30, 5), 0.337290, 0.101534),
                    (ECA(30, 6), 0.231872, 0.069801),
                    (ECA(110, 2), 0.000000, 0.000000),
                    (ECA(110, 3), 0.000000, 0.000000),
                    (ECA(110, 4), 1.561278, 0.469992),
                    (ECA(110, 5), 0.000000, 0.000000),
                    (ECA(110, 6), 1.469012, 0.442217)]

        for net, base2, base10 in networks:
            landscape = Landscape(net)
            self.assertAlmostEqual(base2, landscape.basin_entropy(), places=6)
            self.assertAlmostEqual(
                base2, landscape.basin_entropy(base=2), places=6)
            self.assertAlmostEqual(
                base10, landscape.basin_entropy(base=10), places=6)

    def test_basin_entropy_wtnetwork(self):
        networks = [(s_pombe, 1.221889, 0.367825),
                    (s_cerevisiae, 0.783858, 0.235965),
                    (c_elegans, 0.854267, 0.257160)]

        for net, base2, base10 in networks:
            landscape = Landscape(net)
            self.assertAlmostEqual(base2, landscape.basin_entropy(), places=6)
            self.assertAlmostEqual(
                base2, landscape.basin_entropy(base=2), places=6)
            self.assertAlmostEqual(
                base10, landscape.basin_entropy(base=10), places=6)

    def test_graph_eca(self):
        for size in range(2, 7):
            landscape = Landscape(ECA(30, size))
            g = landscape.graph
            self.assertEqual(landscape.volume, g.number_of_nodes())
            self.assertEqual(landscape.volume, g.number_of_edges())

    def test_graph_wtnetworks(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            g = landscape.graph
            self.assertEqual(landscape.volume, g.number_of_nodes())
            self.assertEqual(landscape.volume, g.number_of_edges())

    def test_in_degree(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = Landscape(ECA(code, size))
                in_degrees = np.empty(landscape.volume, dtype=np.int)
                for i in range(landscape.volume):
                    in_degrees[i] = np.count_nonzero(
                        landscape.transitions == i)
                self.assertEqual(list(in_degrees), list(landscape.in_degrees))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            in_degrees = np.empty(landscape.volume, dtype=np.int)
            for i in range(landscape.volume):
                in_degrees[i] = np.count_nonzero(landscape.transitions == i)
            self.assertEqual(list(in_degrees), list(landscape.in_degrees))

    def test_heights(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = Landscape(ECA(code, size))
                heights = [0] * landscape.volume
                for i in range(landscape.volume):
                    b = landscape.basins[i]
                    state = i
                    while state not in landscape.attractors[b]:
                        heights[i] += 1
                        state = landscape.transitions[state]
                self.assertEqual(heights, list(landscape.heights))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            heights = [0] * landscape.volume
            for i in range(landscape.volume):
                b = landscape.basins[i]
                state = i
                while state not in landscape.attractors[b]:
                    heights[i] += 1
                    state = landscape.transitions[state]
            self.assertEqual(heights, list(landscape.heights))

    def test_attractor_lengths(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = Landscape(ECA(code, size=size))
                lengths = list(map(len, landscape.attractors))
                self.assertEqual(lengths, list(landscape.attractor_lengths))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            lengths = list(map(len, landscape.attractors))
            self.assertEqual(lengths, list(landscape.attractor_lengths))

    def test_recurrence_times(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = Landscape(ECA(code, size))
                recurrence_times = [0] * landscape.volume
                for i in range(landscape.volume):
                    b = landscape.basins[i]
                    recurrence_times[i] = landscape.heights[i] + \
                        landscape.attractor_lengths[b] - 1
                self.assertEqual(recurrence_times, list(
                    landscape.recurrence_times))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = Landscape(net)
            recurrence_times = [0] * landscape.volume
            for i in range(landscape.volume):
                b = landscape.basins[i]
                recurrence_times[i] = landscape.heights[i] + \
                    landscape.attractor_lengths[b] - 1
            self.assertEqual(recurrence_times, list(
                landscape.recurrence_times))
