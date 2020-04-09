from neet.boolean.logicnetwork import LogicNetwork
from neet.boolean.examples import myeloid, s_pombe
from neet.boolean import random
import networkx as nx
import unittest

class TestConstraints(unittest.TestCase):
    #Topological constraints test cases
    def test_negative_external_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasExternalNodes, -1)
    def test_int_external_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasExternalNodes, 0.6)
    def test_has_zero_external_nodes(self):
        #s_pombe has 0 external nodes, check that satisfies returns true for this
        self.assertEqual(random.constraints.HasExternalNodes(0).satisfies(s_pombe.network_graph()), True)
    def test_has_nonzero_external_nodes(self):
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        self.assertEqual(random.constraints.HasExternalNodes(0).satisfies(g), False)
    def test_is_connected_true(self):
        #s_pombe is a network that is weakly connected
        self.assertEqual(random.constraints.IsConnected().satisfies(s_pombe.network_graph()), True)
    def test_is_connected_false(self):
        g = nx.DiGraph([(0,1), (1, 0), (2, 3), (3, 2)])
        self.assertEqual(random.constraints.IsConnected().satisfies(g), False)

    #GenericTopological test cases
    #def test_generic_topological(self):

    #--------------------------------
    #Dynamical Constraints Test Cases:
    def test_is_irreducible_not_implemented_error(self):
        #check that a not implemented error is raised if network passed in is not a logic network
        self.assertRaises(NotImplementedError, random.constraints.IsIrreducible().satisfies, s_pombe)
    def test_is_irreducible_true(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0, 1), {'11'})])
        self.assertEqual(random.constraints.IsIrreducible().satisfies(net), True)
    def test_is_irreducible_false(self):
        net = LogicNetwork([((1, 2), {'01', '10'}),
                            ((0, 2), ('01', '10', '11')),
                            ((0,), {'1', '0'})])
        self.assertEqual(random.constraints.IsIrreducible().satisfies(net), False)
    def test_negative_canalizing_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasCanalizingNodes, -1)
    def test_int_canalizing_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasCanalizingNodes, 0.6)
    def test_canalizing_nodes_false(self):
        #s_pombe has 5 canalizing nodes and myeloid has 11
        self.assertEqual(random.constraints.HasCanalizingNodes(0).satisfies(myeloid), False)
        self.assertEqual(random.constraints.HasCanalizingNodes(0).satisfies(s_pombe), False)
    def test_canalizing_nodes_true(self):
        #s_pombe has 5 canalizing nodes and myeloid has 11
        self.assertEqual(random.constraints.HasCanalizingNodes(11).satisfies(myeloid), True)
        self.assertEqual(random.constraints.HasCanalizingNodes(5).satisfies(s_pombe), True)            
    #GenericDynamical test cases