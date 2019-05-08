import unittest
from neet.boolean import ECA, WTNetwork, LogicNetwork
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
import numpy as np


class TestLandscape(unittest.TestCase):
    """
    Test the ``neet.synchronous.Landscape`` class
    """

    def test_canary(self):
        """
        Canary test
        """
        self.assertTrue(True)

    def test_transitions_eca(self):
        ca = ECA(30, 1)

        landscape = ca.landscape()
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(1, landscape.size)
        self.assertEqual([0, 0], list(landscape.transitions))

        ca.size = 2
        landscape = ca.landscape()
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 0], list(landscape.transitions))

        ca.size = 3
        landscape = ca.landscape()
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

        ca.size = 10
        landscape = ca.landscape()
        trans = landscape.transitions
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(10, landscape.size)
        self.assertEqual(1024, len(trans))
        self.assertEqual([0, 515, 7, 517, 14, 525, 11,
                          521, 28, 543], list(trans[:10]))
        self.assertEqual([18, 16, 13, 14, 10, 8, 7, 4, 2, 0],
                         list(trans[-10:]))

    def test_transitions_eca_index(self):
        ca = ECA(30, 3)
        landscape = ca.landscape(index=1)
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 3, 2, 1, 6, 5, 6, 5], list(landscape.transitions))

        landscape = ca.landscape(index=0)
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 1, 3, 3, 5, 4, 6, 6], list(landscape.transitions))

        landscape = ca.landscape(index=None)
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_eca_pin(self):
        ca = ECA(30, 3)
        landscape = ca.landscape(pin=[1])
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 5, 7, 3, 5, 4, 2, 2], list(landscape.transitions))

        landscape = ca.landscape(pin=[0])
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 6, 1, 6, 5, 2, 1], list(landscape.transitions))

        landscape = ca.landscape(pin=None)
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_eca_values(self):
        ca = ECA(30, 3)
        landscape = ca.landscape(values={0: 1})
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([1, 7, 7, 1, 7, 5, 3, 1], list(landscape.transitions))

        landscape = ca.landscape(values={1: 0})
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 5, 5, 1, 5, 4, 0, 0], list(landscape.transitions))

        landscape = ca.landscape(values={})
        # self.assertEqual(ca, landscape.network)
        self.assertEqual(3, landscape.size)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(landscape.transitions))

    def test_transitions_wtnetwork(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = net.landscape()
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_index(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = net.landscape(index=1)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        landscape = net.landscape(index=0)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 3], list(landscape.transitions))

        landscape = net.landscape(index=None)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_pin(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = net.landscape(pin=[1])
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 2, 3], list(landscape.transitions))

        landscape = net.landscape(pin=[0])
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        landscape = net.landscape(pin=None)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

    def test_transitions_wtnetwork_values(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        landscape = net.landscape(values={0: 1})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([3, 1, 3, 3], list(landscape.transitions))

        landscape = net.landscape(values={1: 0})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 1, 0, 1], list(landscape.transitions))

        landscape = net.landscape(values={})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([2, 1, 2, 3], list(landscape.transitions))

        # Add more test for different thetas?

    def test_transitions_logicnetwork(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = net.landscape()
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_index(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = net.landscape(index=1)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 3, 0, 3], list(landscape.transitions))

        landscape = net.landscape(index=0)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 3, 3], list(landscape.transitions))

        landscape = net.landscape(index=None)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_pin(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = net.landscape(pin=[1])
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 3, 3], list(landscape.transitions))

        landscape = net.landscape(pin=[0])
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([0, 3, 0, 3], list(landscape.transitions))

        landscape = net.landscape(pin=None)
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

    def test_transitions_logicnetwork_values(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        landscape = net.landscape(values={0: 1})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

        landscape = net.landscape(values={1: 0})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 1, 1, 1], list(landscape.transitions))

        landscape = net.landscape(values={})
        # self.assertEqual(net, landscape.network)
        self.assertEqual(2, landscape.size)
        self.assertEqual([1, 3, 1, 3], list(landscape.transitions))

        # Add more test for different thetas?

    def test_transitions_spombe(self):
        landscape = s_pombe.landscape()
        # self.assertEqual(s_pombe, landscape.network)
        self.assertEqual(9, landscape.size)
        trans = landscape.transitions
        self.assertEqual(512, len(trans))
        self.assertEqual([2, 2, 130, 130, 4, 0, 128,
                          128, 8, 0], list(trans[:10]))
        self.assertEqual([464, 464, 344, 336, 464, 464, 348,
                          336, 464, 464], list(trans[-10:]))

    def test_is_state_space(self):
        # self.assertTrue(issubclass(Landscape, StateSpace))

        landscape = s_pombe.landscape()
        self.assertEqual(list(s_pombe), list(landscape))
        self.assertEqual([s_pombe.encode(state) for state in s_pombe],
                         list(map(landscape.encode, landscape)))

        ca = ECA(30, 10)
        landscape = ca.landscape()
        self.assertEqual(list(ca), list(landscape))
        self.assertEqual([ca.encode(state) for state in ca],
                         list(map(landscape.encode, landscape)))

    def test_attractors_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            landscape = rule.landscape()
            self.assertEqual(size, len(landscape.attractors))

    def test_basins_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            landscape = rule.landscape()
            self.assertEqual(landscape.volume, len(landscape.basins))
            self.assertEqual(size, 1 + np.max(landscape.basins))

            unique = list(set(landscape.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_attractors_wtnetworks(self):
        networks = [(s_pombe, 13), (s_cerevisiae, 7), (c_elegans, 5)]
        for net, size in networks:
            landscape = net.landscape()
            self.assertEqual(size, len(landscape.attractors))
            self.assertEqual(landscape.volume, len(landscape.basins))
            self.assertEqual(size, 1 + np.max(landscape.basins))

            unique = list(set(landscape.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_trajectory_too_short(self):
        landscape = ECA(30, 3).landscape()
        with self.assertRaises(ValueError):
            landscape.trajectory([0, 0, 0], timesteps=-1)

        with self.assertRaises(ValueError):
            landscape.trajectory([0, 0, 0], timesteps=0)

    def test_trajectory_eca(self):
        landscape = ECA(30, 3).landscape()

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

        landscape = net.landscape()

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

        landscape = net.landscape()

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
        landscape = ECA(30, 3).landscape()
        with self.assertRaises(ValueError):
            landscape.timeseries(-1)

        with self.assertRaises(ValueError):
            landscape.timeseries(0)

    def test_timeseries_eca(self):
        for size in [5, 7, 11]:
            rule = ECA(30, size)
            landscape = rule.landscape()
            time = 10
            series = landscape.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(rule):
                traj = landscape.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_timeseries_wtnetworks(self):
        for net, size in [(s_pombe, 9), (s_cerevisiae, 11), (c_elegans, 8)]:
            landscape = net.landscape()
            time = 10
            series = landscape.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(net):
                traj = landscape.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_basin_sizes(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
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
            landscape = net.landscape()
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
            landscape = net.landscape()
            self.assertAlmostEqual(base2, landscape.basin_entropy(), places=6)
            self.assertAlmostEqual(
                base2, landscape.basin_entropy(base=2), places=6)
            self.assertAlmostEqual(
                base10, landscape.basin_entropy(base=10), places=6)

    def test_graph_eca(self):
        for size in range(2, 7):
            landscape = ECA(30, size).landscape()
            g = landscape.transition_graph
            self.assertEqual(landscape.volume, g.number_of_nodes())
            self.assertEqual(landscape.volume, g.number_of_edges())

    def test_graph_wtnetworks(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
            g = landscape.transition_graph
            self.assertEqual(landscape.volume, g.number_of_nodes())
            self.assertEqual(landscape.volume, g.number_of_edges())

    def test_in_degree(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = ECA(code, size).landscape()
                in_degrees = np.empty(landscape.volume, dtype=np.int)
                for i in range(landscape.volume):
                    in_degrees[i] = np.count_nonzero(
                        landscape.transitions == i)
                self.assertEqual(list(in_degrees), list(landscape.in_degrees))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
            in_degrees = np.empty(landscape.volume, dtype=np.int)
            for i in range(landscape.volume):
                in_degrees[i] = np.count_nonzero(landscape.transitions == i)
            self.assertEqual(list(in_degrees), list(landscape.in_degrees))

    def test_heights(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = ECA(code, size).landscape()
                heights = [0] * landscape.volume
                for i in range(landscape.volume):
                    b = landscape.basins[i]
                    state = i
                    while state not in landscape.attractors[b]:
                        heights[i] += 1
                        state = landscape.transitions[state]
                self.assertEqual(heights, list(landscape.heights))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
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
                landscape = ECA(code, size).landscape()
                lengths = list(map(len, landscape.attractors))
                self.assertEqual(lengths, list(landscape.attractor_lengths))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
            lengths = list(map(len, landscape.attractors))
            self.assertEqual(lengths, list(landscape.attractor_lengths))

    def test_recurrence_times(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                landscape = ECA(code, size).landscape()
                recurrence_times = [0] * landscape.volume
                for i in range(landscape.volume):
                    b = landscape.basins[i]
                    recurrence_times[i] = landscape.heights[i] + \
                        landscape.attractor_lengths[b] - 1
                self.assertEqual(recurrence_times, list(
                    landscape.recurrence_times))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            landscape = net.landscape()
            recurrence_times = [0] * landscape.volume
            for i in range(landscape.volume):
                b = landscape.basins[i]
                recurrence_times[i] = landscape.heights[i] + \
                    landscape.attractor_lengths[b] - 1
            self.assertEqual(recurrence_times, list(
                landscape.recurrence_times))
