from neet.boolean.logicnetwork import LogicNetwork
from neet.boolean.examples import myeloid, s_pombe
from neet.boolean import random
import unittest

class TestConstraints(unittest.TestCase):
    #Topological constraints test cases
    #HasExternalNodes test cases
    def test_negative_external_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasExternalNodes, -1)
    def test_int_external_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasExternalNodes, 0.6)
    def test_has_zero_external_nodes(self):
        #s_pombe has 0 external nodes, check that satisfies returns true for this
        self.assertEqual(random.constraints.HasExternalNodes(0).satisfies(s_pombe.network_graph()), True)
        #To do: add test for > 0 external nodes
        
    #IsConnected test cases
    def test_is_connected_true(self):
        #s_pombe is a network that is weakly connected
        self.assertEqual(random.constraints.IsConnected().satisfies(s_pombe.network_graph()), True)
    #def test_is_connected_false(self):

    #GenericTopological test cases
    #def test_generic_topological(self):

    #--------------------------------
    #Dynamical Constraints Test Cases:
    #IsIrreducible test cases
    def test_is_irreducible_not_implemented_error(self):
        #check that a not implemented error is raised if network passed in is not a logic network
        self.assertRaises(NotImplementedError, random.constraints.IsIrreducible().satisfies, s_pombe)
    def test_is_irreducible_true(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'})])
        self.assertEqual(random.constraints.IsIrreducible().satisfies(net), True)
    #def test_is_irreducible_false(self):
        # net = LogicNetwork([((1, 2), {'01', '10'}),
        #                     ((0, 2), ('01', '10', '11')),
        #                     ((0, 1), {'11'})])
        # self.assertEqual(random.constraints.IsIrreducible().satisfies(net), False)
        
    #HasCanalizingNodes test cases
    def test_negative_canalizing_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasCanalizingNodes, -1)
    def test_int_canalizing_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasCanalizingNodes, 0.6)
    #GenericDynamical test cases