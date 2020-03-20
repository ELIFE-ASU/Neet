from copy import deepcopy
from neet.boolean import LogicNetwork, WTNetwork
from neet.boolean.examples import s_pombe, c_elegans
import numpy as np
import unittest
import matplotlib


class TestSensitivity(unittest.TestCase):
    def setUp(self):
        self.net_1 = LogicNetwork([
            ((0, 1, 2), {'011', '101', '100', '110'}),
            ((0, 1, 2), {'001', '010', '101', '110'}),
            ((0, 1, 2), {'000', '010', '100', '110'})
        ])
        self.net_2 = LogicNetwork([
            ((0, 1, 2), {'010', '110', '100', '101'}),
            ((0, 1, 2), {'001', '011', '010', '101'}),
            ((0, 1, 2), {'000', '001', '100', '101'})
        ])
        self.net_3 = LogicNetwork([
            ((0, 1, 2), ['001', '010', '100', '111']),
            ((1,), ['0']),
            ((0, 1), ['11'])
        ], names='ABC')

    def test_sensitivity(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        self.assertEqual(1.0, net.sensitivity([0, 0]))

    def test_sensitivity_transitions(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        trans = list(map(net.decode, net.transitions))
        self.assertEqual(1.0, net.sensitivity([0, 0], transitions=trans))

    def test_average_sensitivity_lengths(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])

        with self.assertRaises(ValueError):
            net.average_sensitivity(states=[[0, 0], [0, 1]], weights=[0, 1, 2])

    def test_different_matrix_without_trans(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        trans = list(map(net.decode, net.transitions))
        for state in net:
            with_trans = net.difference_matrix(state[:], transitions=trans)
            without_trans = net.difference_matrix(state[:])
            self.assertTrue(np.allclose(with_trans, without_trans, atol=1e-6))

    def test_average_difference_matrix_without_calc(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        states = list(net)
        with_calc = net.average_sensitivity(states=deepcopy(states), calc_trans=True)
        without_calc = net.average_sensitivity(states=deepcopy(states), calc_trans=False)
        self.assertTrue(np.allclose(with_calc, without_calc, atol=1e-6))

    def test_average_sensitivity(self):
        net = WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        self.assertEqual(1.0, net.average_sensitivity())

    def test_sensitivity_s_pombe(self):
        s = s_pombe.sensitivity([0, 0, 0, 0, 0, 1, 1, 0, 0])
        self.assertAlmostEqual(s, 1.0)

    def test_average_sensitivity_c_elegans(self):
        from neet.boolean.examples import c_elegans

        s = c_elegans.average_sensitivity()
        self.assertAlmostEqual(s, 1.265625)

        s = c_elegans.average_sensitivity(
            states=[[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1]],
            weights=[9, 1])
        self.assertAlmostEqual(s, 1.7)

    def test_lambdaQ_c_elegans(self):
        from neet.boolean.examples import c_elegans
        self.assertAlmostEqual(c_elegans.lambdaQ(), 1.263099227661824)

    def test_average_sensitivity_logic_network(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'})])

        s = net.average_sensitivity()
        self.assertAlmostEqual(s, 1.3333333333333333)

        s = net.average_sensitivity(weights=np.ones(8))
        self.assertAlmostEqual(s, 1.3333333333333333)

        s = net.average_sensitivity(states=list(net))
        self.assertAlmostEqual(s, 1.3333333333333333)

    def test_lambdaQ_logic_network(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'})])
        self.assertAlmostEqual(net.lambdaQ(), 1.2807764064044149)

    def test_is_canalizing_logic_network(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'}),
                            ((3,), {'0'})])

        self.assertFalse(net.is_canalizing(0, 1))
        self.assertTrue(net.is_canalizing(1, 0))
        self.assertTrue(net.is_canalizing(2, 1))
        self.assertFalse(net.is_canalizing(0, 3))
        self.assertFalse(net.is_canalizing(3, 0))

    def test_is_canalizing_wtnetwork(self):
        net = WTNetwork([[0, 1], [1, 0]], [1, 0.5],
                        theta=WTNetwork.negative_threshold)
        self.assertTrue(net.is_canalizing(1, 0))
        self.assertFalse(net.is_canalizing(0, 0))

    def test_canalizing(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'})])

        edges = net.canalizing_edges()
        self.assertEqual(edges, {(1, 0), (1, 2), (2, 0), (2, 1)})

        nodes = net.canalizing_nodes()
        self.assertEqual(nodes, {1, 2})

    def test_average_sensitivity_hgf(self):
        from neet.boolean.examples import hgf_signaling_in_keratinocytes
        self.assertAlmostEqual(hgf_signaling_in_keratinocytes.average_sensitivity(),
                               0.981618, places=6)

    def test_average_sensitivity_il_6(self):
        from neet.boolean.examples import il_6_signaling
        self.assertAlmostEqual(il_6_signaling.average_sensitivity(), 0.914971, places=6)

    def test_time_sensitivity_noninteger_timestep(self):
        with self.assertRaises(TypeError):
            s_pombe.sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], timesteps=2.5)

        with self.assertRaises(TypeError):
            s_pombe.average_sensitivity(timesteps='a')

    def test_time_sensitivity_negative_timestep(self):
        with self.assertRaises(ValueError):
            s_pombe.sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], timesteps=-1)

        with self.assertRaises(ValueError):
            s_pombe.average_sensitivity(timesteps=-2)

    def test_time_sensitivity_unchanged_state(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        s_pombe.sensitivity(state, timesteps=2)
        self.assertEqual(state, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_time_sensitivity_s_pombe(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=2), 1.777778, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=3), 2.111111, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=4), 2.666667, places=6)

        state = [0, 1, 1, 0, 1, 1, 0, 1, 1]
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=2), 0.555556, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=3), 0.222222, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=4), 0.555556, places=6)

        state = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=2), 1.333333, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=3), 0.555556, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity(state, timesteps=4), 1.888889, places=6)

    def test_time_sensitivity_c_elegans(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=2), 2.000000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=3), 2.250000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=4), 2.375000, places=6)

        state = [0, 1, 1, 0, 0, 1, 1, 0]
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=2), 0.750000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=3), 0.750000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=4), 0.625000, places=6)

        state = [0, 0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=2), 2.750000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=3), 3.250000, places=6)
        self.assertAlmostEqual(c_elegans.sensitivity(state, timesteps=4), 2.875000, places=6)

    def test_average_time_sensitivity(self):
        self.assertAlmostEqual(self.net_1.average_sensitivity(timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(self.net_1.average_sensitivity(timesteps=2), 1.333333, places=6)
        self.assertAlmostEqual(self.net_1.average_sensitivity(timesteps=3), 1.666667, places=6)
        self.assertAlmostEqual(self.net_1.average_sensitivity(timesteps=4), 1.000000, places=6)

        self.assertAlmostEqual(self.net_2.average_sensitivity(timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(self.net_2.average_sensitivity(timesteps=2), 1.500000, places=6)
        self.assertAlmostEqual(self.net_2.average_sensitivity(timesteps=3), 1.666667, places=6)
        self.assertAlmostEqual(self.net_2.average_sensitivity(timesteps=4), 1.666667, places=6)

        self.assertAlmostEqual(self.net_3.average_sensitivity(timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(self.net_3.average_sensitivity(timesteps=2), 1.500000, places=6)
        self.assertAlmostEqual(self.net_3.average_sensitivity(timesteps=3), 1.000000, places=6)
        self.assertAlmostEqual(self.net_3.average_sensitivity(timesteps=4), 1.000000, places=6)

        states = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1, 0, 0, 1]]
        self.assertAlmostEqual(s_pombe.average_sensitivity(states, timesteps=0), 1.000000, places=6)
        self.assertAlmostEqual(s_pombe.average_sensitivity(states, timesteps=2), 1.222222, places=6)
        self.assertAlmostEqual(s_pombe.average_sensitivity(states, timesteps=3), 0.962963, places=6)
        self.assertAlmostEqual(s_pombe.average_sensitivity(states, timesteps=4), 1.703704, places=6)

    def test_c_sensitivity_noninteger_timestep(self):
        with self.assertRaises(TypeError):
            s_pombe.c_sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], c=2.5)

        with self.assertRaises(TypeError):
            s_pombe.average_c_sensitivity(c='a')

    def test_c_sensitivity_negative_timestep(self):
        with self.assertRaises(ValueError):
            s_pombe.c_sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], c=-1)

        with self.assertRaises(ValueError):
            s_pombe.average_c_sensitivity(c=-2)

    def test_c_sensitivity_large_perturbation(self):
        with self.assertRaises(ValueError):
            s_pombe.c_sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], c=10)

        with self.assertRaises(ValueError):
            s_pombe.average_c_sensitivity([0, 0, 0, 0, 0, 0, 0, 0, 0], c=10)

    def test_c_sensitivity_unchanged_state(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        s_pombe.c_sensitivity(state, c=2)
        self.assertEqual(state, [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_c_sensitivity_c1(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(s_pombe.c_sensitivity(state, c=1), s_pombe.sensitivity(state))

        state = [0, 1, 1, 0, 1, 1, 0, 1, 1]
        self.assertEqual(s_pombe.c_sensitivity(state, c=1), s_pombe.sensitivity(state))

        state = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        self.assertEqual(s_pombe.c_sensitivity(state, c=1), s_pombe.sensitivity(state))

    def test_c_sensitivity_s_pombe(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=2), 2.361111, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=7), 4.000000, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=8), 4.333333, places=6)

        state = [0, 1, 1, 0, 1, 1, 0, 1, 1]
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=2), 1.555556, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=7), 3.333333, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=8), 3.666667, places=6)

        state = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=2), 1.944444, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=7), 4.222222, places=6)
        self.assertAlmostEqual(s_pombe.c_sensitivity(state, c=8), 4.555556, places=6)

    def test_c_sensitivity_c_elegans(self):
        state = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=2), 2.535714, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=7), 4.750000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=8), 5.000000, places=6)

        state = [0, 1, 1, 0, 0, 1, 1, 0]
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=2), 1.464286, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=7), 5.875000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=8), 7.000000, places=6)

        state = [0, 0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=0), 0.000000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=2), 2.928571, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=7), 5.750000, places=6)
        self.assertAlmostEqual(c_elegans.c_sensitivity(state, c=8), 6.000000, places=6)

    def test_average_c_sensitivity(self):
        self.assertAlmostEqual(self.net_1.average_c_sensitivity(c=0), 0.000000, places=6)
        self.assertAlmostEqual(self.net_1.average_c_sensitivity(c=2), 1.833333, places=6)

        self.assertAlmostEqual(self.net_2.average_c_sensitivity(c=0), 0.000000, places=6)
        self.assertAlmostEqual(self.net_2.average_c_sensitivity(c=2), 2.000000, places=6)

        self.assertAlmostEqual(self.net_3.average_c_sensitivity(c=0), 0.000000, places=6)
        self.assertAlmostEqual(self.net_3.average_c_sensitivity(c=2), 1.166667, places=6)

    def test_average_c_sensitivity_no_trans(self):
        self.assertAlmostEqual(self.net_1.average_c_sensitivity(c=0, calc_trans=False),
                               0.000000, places=6)
        self.assertAlmostEqual(self.net_1.average_c_sensitivity(c=2, calc_trans=False),
                               1.833333, places=6)

        self.assertAlmostEqual(self.net_2.average_c_sensitivity(c=0, calc_trans=False),
                               0.000000, places=6)
        self.assertAlmostEqual(self.net_2.average_c_sensitivity(c=2, calc_trans=False),
                               2.000000, places=6)

        self.assertAlmostEqual(self.net_3.average_c_sensitivity(c=0, calc_trans=False),
                               0.000000, places=6)
        self.assertAlmostEqual(self.net_3.average_c_sensitivity(c=2, calc_trans=False),
                               1.166667, places=6)

    def test_average_c_sensitivity_with_states(self):
        states = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1, 0, 0, 1]]

        self.assertAlmostEqual(s_pombe.average_c_sensitivity(states, c=0), 0.000000, places=6)
        self.assertAlmostEqual(s_pombe.average_c_sensitivity(states, c=2), 1.953704, places=6)
        self.assertAlmostEqual(s_pombe.average_c_sensitivity(states, c=7), 3.851852, places=6)
        self.assertAlmostEqual(s_pombe.average_c_sensitivity(states, c=8), 4.185185, places=6)

    def test_derrida_plot_raises(self):
        with self.assertRaises(TypeError):
            s_pombe.derrida_plot(min_c='a')

        with self.assertRaises(ValueError):
            s_pombe.derrida_plot(min_c=-1)

        with self.assertRaises(ValueError):
            s_pombe.derrida_plot(min_c=s_pombe.size + 1)

        with self.assertRaises(TypeError):
            s_pombe.derrida_plot(max_c='a')

        with self.assertRaises(ValueError):
            s_pombe.derrida_plot(max_c=-1)

        with self.assertRaises(ValueError):
            s_pombe.derrida_plot(max_c=s_pombe.size + 2)

        with self.assertRaises(ValueError):
            s_pombe.derrida_plot(min_c=5, max_c=3)

    def test_derrida_plot_max_c(self):
        f, ax = self.net_1.derrida_plot(min_c=1, max_c=3)
        self.assertIsInstance(f, matplotlib.figure.Figure)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(list(ax.lines[0].get_xdata()), [1, 2])
        self.assertTrue(np.allclose(ax.lines[0].get_ydata(), [1.666667, 1.833333], atol=1e-6))

    def test_derrida_plot_no_max(self):
        f, ax = self.net_1.derrida_plot()
        self.assertIsInstance(f, matplotlib.figure.Figure)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(list(ax.lines[0].get_xdata()), [0, 1, 2, 3])
        self.assertTrue(np.allclose(ax.lines[0].get_ydata(),
                                    [0.000000, 1.666667, 1.833333, 1.500000], atol=1e-6))

    def test_extended_time_plot_raises(self):
        with self.assertRaises(TypeError):
            s_pombe.extended_time_plot(min_timesteps='a')

        with self.assertRaises(ValueError):
            s_pombe.extended_time_plot(min_timesteps=-1)

        with self.assertRaises(TypeError):
            s_pombe.extended_time_plot(max_timesteps='a')

        with self.assertRaises(ValueError):
            s_pombe.extended_time_plot(max_timesteps=-1)

        with self.assertRaises(ValueError):
            s_pombe.extended_time_plot(min_timesteps=5, max_timesteps=3)

    def test_extended_time_plot(self):
        f, ax = self.net_1.extended_time_plot(min_timesteps=2, max_timesteps=5)
        self.assertIsInstance(f, matplotlib.figure.Figure)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(list(ax.lines[0].get_xdata()), [2, 3, 4])
        self.assertTrue(np.allclose(ax.lines[0].get_ydata(),
                                    [1.333333, 1.666667, 1.000000], atol=1e-6))
