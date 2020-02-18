from copy import deepcopy
from neet.boolean import LogicNetwork, WTNetwork
import numpy as np
import unittest


class TestSensitivity(unittest.TestCase):
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
        from neet.boolean.examples import s_pombe
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

    def test_time_sensitivity_s_pombe(self):
        from neet.boolean.examples import s_pombe
        #s_pombe sensitivity for timesteps = 2-4 for 3 different states
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,0,0,0,0,0,0,0], timesteps=2), 1.777778, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,0,0,0,0,0,0,0], timesteps=3), 2.111111, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,0,0,0,0,0,0,0], timesteps=4), 2.666667, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,1,1,0,1,1,0,1,1], timesteps=2), 0.555556, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,1,1,0,1,1,0,1,1], timesteps=3), 0.222222, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,1,1,0,1,1,0,1,1], timesteps=4), 0.555556, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,1,0,0,1,0,0,1], timesteps=2), 1.333333, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,1,0,0,1,0,0,1], timesteps=3), 0.555556, places=6)
        self.assertAlmostEqual(s_pombe.sensitivity([0,0,1,0,0,1,0,0,1], timesteps=4), 1.888889, places=6)

    def test_time_sensitivity_c_elegans(self):
        from neet.boolean.examples import c_elegans
        #c_elegans sensitivity for timesteps = 2-4 for 3 different states
        self.assertEqual(c_elegans.sensitivity([0,0,0,0,0,0,0,0], timesteps=2), 2)
        self.assertEqual(c_elegans.sensitivity([0,0,0,0,0,0,0,0], timesteps=3), 2.25)
        self.assertEqual(c_elegans.sensitivity([0,0,0,0,0,0,0,0], timesteps=4), 2.375)
        self.assertEqual(c_elegans.sensitivity([0,1,1,0,0,1,1,0], timesteps=2), 0.75)
        self.assertEqual(c_elegans.sensitivity([0,1,1,0,0,1,1,0], timesteps=3), 0.75)
        self.assertEqual(c_elegans.sensitivity([0,1,1,0,0,1,1,0], timesteps=4), 0.625)
        self.assertEqual(c_elegans.sensitivity([0,0,0,1,0,0,0,1], timesteps=2), 2.75)
        self.assertEqual(c_elegans.sensitivity([0,0,0,1,0,0,0,1], timesteps=3), 3.25)
        self.assertEqual(c_elegans.sensitivity([0,0,0,1,0,0,0,1], timesteps=4), 2.875)
    
