import unittest
from neet.boolean import ECA, WTNetwork, LogicNetwork
from neet.boolean.examples import s_pombe, s_cerevisiae, c_elegans
import numpy as np


class TestLandscape(unittest.TestCase):
    """
    Test the ``neet.landscape.Landscape`` class
    """

    def test_canary(self):
        """
        Canary test
        """
        self.assertTrue(True)

    def tearDown(self):
        s_pombe.clear_landscape()
        s_cerevisiae.clear_landscape()
        c_elegans.clear_landscape()

    def test_transitions_eca(self):
        net = ECA(30, 1)
        self.assertEqual([0, 0], list(net.transitions))

        net.size = 2
        self.assertEqual([0, 1, 2, 0], list(net.transitions))

        net.size = 3
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(net.transitions))

        net.size = 10
        trans = net.transitions
        self.assertEqual(1024, len(trans))
        self.assertEqual([0, 515, 7, 517, 14, 525, 11,
                          521, 28, 543], list(trans[:10]))
        self.assertEqual([18, 16, 13, 14, 10, 8, 7, 4, 2, 0],
                         list(trans[-10:]))

    def test_transitions_eca_index(self):
        net = ECA(30, 3)
        net.landscape(index=1)
        self.assertEqual([0, 3, 2, 1, 6, 5, 6, 5], list(net.transitions))

        net.landscape(index=0)
        self.assertEqual([0, 1, 3, 3, 5, 4, 6, 6], list(net.transitions))

        net.landscape(index=None)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(net.transitions))

    def test_transitions_eca_pin(self):
        net = ECA(30, 3)
        net.landscape(pin=[1])
        self.assertEqual([0, 5, 7, 3, 5, 4, 2, 2], list(net.transitions))

        net.landscape(pin=[0])
        self.assertEqual([0, 7, 6, 1, 6, 5, 2, 1], list(net.transitions))

        net.landscape(pin=None)
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(net.transitions))

    def test_transitions_eca_values(self):
        net = ECA(30, 3)
        net.landscape(values={0: 1})
        self.assertEqual([1, 7, 7, 1, 7, 5, 3, 1], list(net.transitions))

        net.landscape(values={1: 0})
        self.assertEqual([0, 5, 5, 1, 5, 4, 0, 0], list(net.transitions))

        net.landscape(values={})
        self.assertEqual([0, 7, 7, 1, 7, 4, 2, 0], list(net.transitions))

    def test_transitions_wtnetwork(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        self.assertEqual([2, 1, 2, 3], list(net.transitions))

    def test_transitions_wtnetwork_index(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        net.landscape(index=1)
        self.assertEqual([2, 1, 2, 3], list(net.transitions))

        net.landscape(index=0)
        self.assertEqual([0, 1, 2, 3], list(net.transitions))

        net.landscape(index=None)
        self.assertEqual([2, 1, 2, 3], list(net.transitions))

    def test_transitions_wtnetwork_pin(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        net.landscape(pin=[1])
        self.assertEqual([0, 1, 2, 3], list(net.transitions))

        net.landscape(pin=[0])
        self.assertEqual([2, 1, 2, 3], list(net.transitions))

        net.landscape(pin=None)
        self.assertEqual([2, 1, 2, 3], list(net.transitions))

    def test_transitions_wtnetwork_values(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 1]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        net.landscape(values={0: 1})
        self.assertEqual([3, 1, 3, 3], list(net.transitions))

        net.landscape(values={1: 0})
        self.assertEqual([0, 1, 0, 1], list(net.transitions))

        net.landscape(values={})
        self.assertEqual([2, 1, 2, 3], list(net.transitions))

    def test_transitions_logicnetwork(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        self.assertEqual([1, 3, 1, 3], list(net.transitions))

    def test_transitions_logicnetwork_index(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        net.landscape(index=1)
        self.assertEqual([0, 3, 0, 3], list(net.transitions))

        net.landscape(index=0)
        self.assertEqual([1, 1, 3, 3], list(net.transitions))

        net.landscape(index=None)
        self.assertEqual([1, 3, 1, 3], list(net.transitions))

    def test_transitions_logicnetwork_pin(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        net.landscape(pin=[1])
        self.assertEqual([1, 1, 3, 3], list(net.transitions))

        net.landscape(pin=[0])
        self.assertEqual([0, 3, 0, 3], list(net.transitions))

        net.landscape(pin=None)
        self.assertEqual([1, 3, 1, 3], list(net.transitions))

    def test_transitions_logicnetwork_values(self):
        net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])

        net.landscape(values={0: 1})
        self.assertEqual([1, 3, 1, 3], list(net.transitions))

        net.landscape(values={1: 0})
        self.assertEqual([1, 1, 1, 1], list(net.transitions))

        net.landscape(values={})
        self.assertEqual([1, 3, 1, 3], list(net.transitions))

        # Add more test for different thetas?

    def test_transitions_spombe(self):
        trans = s_pombe.transitions
        self.assertEqual(512, len(trans))
        self.assertEqual([2, 2, 130, 130, 4, 0, 128,
                          128, 8, 0], list(trans[:10]))
        self.assertEqual([464, 464, 344, 336, 464, 464, 348,
                          336, 464, 464], list(trans[-10:]))

    def test_attractors_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            self.assertEqual(size, len(rule.attractors))

    def test_basins_eca(self):
        networks = [(ECA(30, 2), 3), (ECA(30, 3), 1), (ECA(30, 4), 4),
                    (ECA(30, 5), 2), (ECA(30, 6), 3), (ECA(110, 2), 1),
                    (ECA(110, 3), 1), (ECA(110, 4), 3), (ECA(110, 5), 1),
                    (ECA(110, 6), 3)]

        for rule, size in networks:
            self.assertEqual(rule.volume, len(rule.basins))
            self.assertEqual(size, 1 + np.max(rule.basins))

            unique = list(set(rule.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_attractors_wtnetworks(self):
        networks = [(s_pombe, 13), (s_cerevisiae, 7), (c_elegans, 5)]
        for net, size in networks:
            self.assertEqual(size, len(net.attractors))
            self.assertEqual(net.volume, len(net.basins))
            self.assertEqual(size, 1 + np.max(net.basins))

            unique = list(set(net.basins))
            unique.sort()
            self.assertEqual(list(range(size)), unique)

    def test_trajectory_too_short(self):
        net = ECA(30, 3)
        with self.assertRaises(ValueError):
            net.trajectory([0, 0, 0], timesteps=-1)

        with self.assertRaises(ValueError):
            net.trajectory([0, 0, 0], timesteps=0)

    def test_trajectory_eca(self):
        net = ECA(30, 3)

        with self.assertRaises(ValueError):
            net.trajectory([])

        with self.assertRaises(ValueError):
            net.trajectory([0, 1])

        xs = [0, 1, 0]

        got = net.trajectory(xs)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = net.trajectory(xs, timesteps=5)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

        got = net.trajectory(xs, encode=False)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = net.trajectory(xs, timesteps=5, encode=False)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

        got = net.trajectory(xs, encode=True)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([2, 7, 0], list(got))

        got = net.trajectory(xs, timesteps=5, encode=True)
        self.assertEqual([0, 1, 0], xs)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        xs = 2
        got = net.trajectory(xs)
        self.assertEqual([2, 7, 0], list(got))

        got = net.trajectory(xs, timesteps=5)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        got = net.trajectory(xs, encode=True)
        self.assertEqual([2, 7, 0], list(got))

        got = net.trajectory(xs, timesteps=5, encode=True)
        self.assertEqual([2, 7, 0, 0, 0, 0], list(got))

        got = net.trajectory(xs, encode=False)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0]], got)

        got = net.trajectory(xs, timesteps=5, encode=False)
        self.assertEqual([[0, 1, 0], [1, 1, 1], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0]], got)

    def test_trajectory_wtnetwork(self):
        net = WTNetwork(
            weights=[[1, 0], [-1, 0]],
            thresholds=[0.5, 0.0],
            theta=WTNetwork.positive_threshold
        )

        state = [0, 0]
        got = net.trajectory(state)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = net.trajectory(state, timesteps=3)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

        got = net.trajectory(state, encode=False)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = net.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([0, 0], state)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

        got = net.trajectory(state, encode=True)
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2], got)

        got = net.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 0], state)
        self.assertEqual([0, 2, 2, 2], got)

        state = 0
        got = net.trajectory(state)
        self.assertEqual([0, 2], got)

        got = net.trajectory(state, timesteps=3)
        self.assertEqual([0, 2, 2, 2], got)

        got = net.trajectory(state, encode=True)
        self.assertEqual([0, 2], got)

        got = net.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 2, 2, 2], got)

        got = net.trajectory(state, encode=False)
        self.assertEqual([[0, 0], [0, 1]], got)

        got = net.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([[0, 0], [0, 1], [0, 1], [0, 1]], got)

    def test_trajectory_logicnetwork(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), {'01', '10', '11'}),
                            ((0, 1), {'11'})])

        state = [0, 1, 0]
        got = net.trajectory(state, encode=False)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = net.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

        got = net.trajectory(state)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = net.trajectory(state, timesteps=3)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

        got = net.trajectory(state, encode=True)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 1], got)

        got = net.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([0, 1, 0], state)
        self.assertEqual([2, 1, 2, 1], got)

        state = 2
        got = net.trajectory(state)
        self.assertEqual([2, 1], got)

        got = net.trajectory(state, timesteps=3)
        self.assertEqual([2, 1, 2, 1], got)

        got = net.trajectory(state, encode=True)
        self.assertEqual([2, 1], got)

        got = net.trajectory(state, timesteps=3, encode=True)
        self.assertEqual([2, 1, 2, 1], got)

        got = net.trajectory(state, encode=False)
        self.assertEqual([[0, 1, 0], [1, 0, 0]], got)

        got = net.trajectory(state, timesteps=3, encode=False)
        self.assertEqual([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], got)

    def test_timeseries_too_short(self):
        net = ECA(30, 3)

        with self.assertRaises(ValueError):
            net.timeseries(-1)

        with self.assertRaises(ValueError):
            net.timeseries(0)

    def test_timeseries_eca(self):
        for size in [5, 7, 11]:
            rule = ECA(30, size)
            time = 10
            series = rule.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(rule):
                traj = rule.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_timeseries_wtnetworks(self):
        for net, size in [(s_pombe, 9), (s_cerevisiae, 11), (c_elegans, 8)]:
            time = 10
            series = net.timeseries(time)
            self.assertEqual((size, 2**size, time + 1), series.shape)
            for index, state in enumerate(net):
                traj = net.trajectory(state, timesteps=time)
                for t, expect in enumerate(traj):
                    got = series[:, index, t]
                    self.assertTrue(np.array_equal(expect, got))

    def test_basin_sizes(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            basins = net.basins
            histogram = [0] * (np.max(basins) + 1)
            for b in basins:
                histogram[b] += 1
            net.clear_landscape()
            self.assertEqual(histogram, list(net.basin_sizes))

    def test_basin_entropy_eca(self):
        networks = [(ECA(30, 2), 1.500000),
                    (ECA(30, 3), 0.000000),
                    (ECA(30, 4), 1.186278),
                    (ECA(30, 5), 0.337290),
                    (ECA(30, 6), 0.231872),
                    (ECA(110, 2), 0.000000),
                    (ECA(110, 3), 0.000000),
                    (ECA(110, 4), 1.561278),
                    (ECA(110, 5), 0.000000),
                    (ECA(110, 6), 1.469012)]

        for net, entropy in networks:
            self.assertAlmostEqual(entropy, net.basin_entropy, places=6)

    def test_basin_entropy_wtnetwork(self):
        networks = [(s_pombe, 1.221889),
                    (s_cerevisiae, 0.783858),
                    (c_elegans, 0.854267)]

        for net, entropy in networks:
            self.assertAlmostEqual(entropy, net.basin_entropy, places=6)

    def test_graph_eca(self):
        for size in range(2, 7):
            net = ECA(30, size)
            g = net.landscape_graph()
            self.assertEqual(net.volume, g.number_of_nodes())
            self.assertEqual(net.volume, g.number_of_edges())

    def test_graph_wtnetworks(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            g = net.landscape_graph()
            self.assertEqual(net.volume, g.number_of_nodes())
            self.assertEqual(net.volume, g.number_of_edges())

    def test_graph_update_eca(self):
        for size in range(2, 7):
            net = ECA(30, size)
            g = net.landscape_graph()
            net.landscape_graph(width=size)
            self.assertEqual(g.graph['width'], size)

    def test_graph_update_wtnetwork(self):
        for net in [s_pombe, s_cerevisiae, c_elegans]:
            net.landscape()
            g = net.landscape_graph()
            net.landscape_graph(label=net.metadata['name'])
            self.assertEqual(g.graph['label'], net.metadata['name'])

    def test_in_degree(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                net = ECA(code, size)
                in_degrees = np.empty(net.volume, dtype=np.int)
                for i in range(net.volume):
                    in_degrees[i] = np.count_nonzero(
                        net.transitions == i)
                net.clear_landscape()
                self.assertEqual(list(in_degrees), list(net.in_degrees))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            in_degrees = np.empty(net.volume, dtype=np.int)
            for i in range(net.volume):
                in_degrees[i] = np.count_nonzero(net.transitions == i)
            net.clear_landscape()
            self.assertEqual(list(in_degrees), list(net.in_degrees))

    def test_heights(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                net = ECA(code, size)
                heights = [0] * net.volume
                for i in range(net.volume):
                    b = net.basins[i]
                    state = i
                    while state not in net.attractors[b]:
                        heights[i] += 1
                        state = net.transitions[state]
                net.clear_landscape()
                self.assertEqual(heights, list(net.heights))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            heights = [0] * net.volume
            for i in range(net.volume):
                b = net.basins[i]
                state = i
                while state not in net.attractors[b]:
                    heights[i] += 1
                    state = net.transitions[state]
            net.clear_landscape()
            self.assertEqual(heights, list(net.heights))

    def test_attractor_lengths(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                net = ECA(code, size)
                lengths = list(map(len, net.attractors))
                net.clear_landscape()
                self.assertEqual(lengths, list(net.attractor_lengths))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            lengths = list(map(len, net.attractors))
            net.clear_landscape()
            self.assertEqual(lengths, list(net.attractor_lengths))

    def test_recurrence_times(self):
        for code in [30, 110, 21, 43]:
            for size in range(2, 7):
                net = ECA(code, size)
                recurrence_times = [0] * net.volume
                for i in range(net.volume):
                    b = net.basins[i]
                    recurrence_times[i] = net.heights[i] + \
                        net.attractor_lengths[b] - 1
                net.clear_landscape()
                self.assertEqual(recurrence_times, list(
                    net.recurrence_times))

        for net in [s_pombe, s_cerevisiae, c_elegans]:
            recurrence_times = [0] * net.volume
            for i in range(net.volume):
                b = net.basins[i]
                recurrence_times[i] = net.heights[i] + \
                    net.attractor_lengths[b] - 1
            net.clear_landscape()
            self.assertEqual(recurrence_times, list(
                net.recurrence_times))

    def test_landscape_data(self):
        net = ECA(30, size=5)
        data_before_setup = net.landscape_data
        self.assertEqual(data_before_setup.__dict__, {})

        data_after_setup = net.landscape().landscape_data
        self.assertEqual(data_before_setup.__dict__, {})
        self.assertEqual(
            list(data_after_setup.__dict__.keys()), ['transitions'])

        data_after_expound = net.expound().landscape_data
        self.assertEqual(data_before_setup.__dict__, {})
        self.assertEqual(set(data_after_setup.__dict__.keys()), set([
            "transitions",
            "basins",
            "basin_sizes",
            "attractors",
            "attractor_lengths",
            "in_degrees",
            "heights",
            "recurrence_times",
            "basin_entropy"
        ]))
        self.assertEqual(set(data_after_expound.__dict__.keys()), set([
            "transitions",
            "basins",
            "basin_sizes",
            "attractors",
            "attractor_lengths",
            "in_degrees",
            "heights",
            "recurrence_times",
            "basin_entropy"
        ]))

    def test_expound(self):
        net = ECA(30, size=5)
        before = net.landscape_data
        after = net.expound().landscape_data
        self.assertEqual(before.__dict__, {})
        self.assertEqual(set(after.__dict__.keys()), set([
            "transitions",
            "basins",
            "basin_sizes",
            "attractors",
            "attractor_lengths",
            "in_degrees",
            "heights",
            "recurrence_times",
            "basin_entropy"
        ]))
