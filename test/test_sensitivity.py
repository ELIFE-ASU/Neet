import unittest
from neet.sensitivity import *
from neet.sensitivity import _hamming_neighbors
from neet.statespace import StateSpace
import neet.boolean as NB


class TestSensitivityWTNetwork(unittest.TestCase):
    class IsBooleanNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace(1)

    class IsNotBooleanNetwork(object):
        def update(self, lattice):
            pass

        def state_space(self):
            return StateSpace([2, 3, 4])

    class IsNotNetwork(object):
        pass

    def test_sensitivity_net_type(self):
        with self.assertRaises(TypeError):
            sensitivity(self.IsNotNetwork(), [0, 0, 0])

        with self.assertRaises(TypeError):
            sensitivity(self.IsNotBooleanNetwork(), [0, 0, 0])

    def test_hamming_neighbors_input(self):
        with self.assertRaises(ValueError):
            _hamming_neighbors([0, 1, 2])

        with self.assertRaises(ValueError):
            _hamming_neighbors([[0, 0, 1], [1, 0, 0]])

    def test_hamming_neighbors_example(self):
        state = [0, 1, 1, 0]
        neighbors = [[1, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 1]]
        self.assertTrue(np.array_equal(neighbors, _hamming_neighbors(state)))

    def test_sensitivity(self):
        net = NB.WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        self.assertEqual(1.0, sensitivity(net, [0, 0]))

    def test_average_sensitivity_net_type(self):
        with self.assertRaises(TypeError):
            average_sensitivity(self.IsNotNetwork())

        with self.assertRaises(TypeError):
            average_sensitivity(self.IsNotBooleanNetwork())

    def test_average_sensitivity_lengths(self):
        net = NB.WTNetwork([[1, -1], [0, 1]], [0.5, 0])

        with self.assertRaises(ValueError):
            average_sensitivity(
                net, states=[[0, 0], [0, 1]], weights=[0, 1, 2])

    def test_average_sensitivity(self):
        net = NB.WTNetwork([[1, -1], [0, 1]], [0.5, 0])
        self.assertEqual(1.0, average_sensitivity(net))

    def test_sensitivity_s_pombe(self):
        from neet.boolean.examples import s_pombe
        s = sensitivity(s_pombe,[0,0,0,0,0,1,1,0,0])
        self.assertAlmostEqual(s,1.0)

    def test_average_sensitivity_c_elegans(self):
        from neet.boolean.examples import c_elegans
        
        s = average_sensitivity(c_elegans)
        self.assertAlmostEqual(s,1.265625)

        s = average_sensitivity(c_elegans,
                                states=[[0,0,0,0,0,0,0,0],
                                        [1,1,1,1,1,1,1,1]],
                                weights=[9,1])
        self.assertAlmostEqual(s,1.7)
    
    def test_lambdaQ_c_elegans(self):
        from neet.boolean.examples import c_elegans
        l = lambdaQ(c_elegans)
        self.assertAlmostEqual(l,1.263099227661824)

    def test_average_sensitivity_logic_network(self):
        net = NB.LogicNetwork([((1, 2), {'01', '10'}),
                               ((0, 2), ('01', '10', '11')),
                               ((0, 1), {'11'})])
        
        s = average_sensitivity(net)
        self.assertAlmostEqual(s,1.3333333333333333)

        s = average_sensitivity(net,weights=np.ones(8))
        self.assertAlmostEqual(s,1.3333333333333333)

        s = average_sensitivity(net,states=net.state_space())
        self.assertAlmostEqual(s,1.3333333333333333)

    def test_lambdaQ_logic_network(self):
        net = NB.LogicNetwork([((1, 2), {'01', '10'}),
                               ((0, 2), ('01', '10', '11')),
                               ((0, 1), {'11'})])
        l = lambdaQ(net)
        self.assertAlmostEqual(l,1.2807764064044149)

    def test_is_canalizing_logic_network(self):
        net = NB.LogicNetwork([((1, 2), {'01', '10'}),
                               ((0, 2), ('01', '10', '11')),
                               ((0, 1), {'11'})])

        self.assertFalse(is_canalizing(net,0,1))
        self.assertTrue(is_canalizing(net,1,0))
        self.assertTrue(is_canalizing(net,2,1))

    def test_canalizing(self):
        net = NB.LogicNetwork([((1, 2), {'01', '10'}),
                               ((0, 2), ('01', '10', '11')),
                               ((0, 1), {'11'})])
        
        edges = canalizing_edges(net)
        self.assertEqual(edges,{(1, 0), (1, 2), (2, 0), (2, 1)})
        
        nodes = canalizing_nodes(net)
        self.assertEqual(nodes,{1,2})

