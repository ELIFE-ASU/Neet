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
    def test_type_error_external_nodes(self):
        #check that you hit a type error if a non-int or non-digraph is passed in
        self.assertRaises(TypeError, random.constraints.HasExternalNodes, s_pombe)
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
    def test_generic_topological_true(self):
        def always_true(self):
            return True
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        self.assertEqual(random.constraints.GenericTopological(always_true).satisfies(g), True)
    def test_generic_topological_false(self):
        def always_false(self):
            return False
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        self.assertEqual(random.constraints.GenericTopological(always_false).satisfies(g), False)
    def test_generic_topological_uncallable(self): 
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        #syntax on this is a little difference because type error returns a message here
        with self.assertRaisesRegex(TypeError, 'test must be callable'):
            random.constraints.GenericTopological(0).satisfies(g)
    def test_generic_topological_graph_type(self):
        def always_true(self):
            return True
        with self.assertRaisesRegex(TypeError, 'only directed graphs are testable with topological constraints'):
            random.constraints.GenericTopological(always_true).satisfies(s_pombe)
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
    def test_type_error_canalizing_nodes(self):
        #check that you hit a type error if a non-int or non-network is passed in
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        self.assertRaises(TypeError, random.constraints.HasCanalizingNodes, 0.6)
        self.assertRaises(TypeError, random.constraints.HasCanalizingNodes(0).satisfies, g)
    def test_canalizing_nodes_false(self):
        #s_pombe has 5 canalizing nodes and myeloid has 11
        self.assertEqual(random.constraints.HasCanalizingNodes(0).satisfies(myeloid), False)
        self.assertEqual(random.constraints.HasCanalizingNodes(0).satisfies(s_pombe), False)
    def test_canalizing_nodes_true(self):
        #s_pombe has 5 canalizing nodes and myeloid has 11
        self.assertEqual(random.constraints.HasCanalizingNodes(11).satisfies(myeloid), True)
        self.assertEqual(random.constraints.HasCanalizingNodes(5).satisfies(s_pombe), True)   
    #GenericDynamical test cases
    def test_generic_dymnamical_true(self):
        def always_true(self):
            return True
        self.assertEqual(random.constraints.GenericDynamical(always_true).satisfies(s_pombe), True)
    def test_generic_dynamical_false(self):
        def always_false(self):
            return False
        self.assertEqual(random.constraints.GenericDynamical(always_false).satisfies(s_pombe), False)
    def test_generic_dynamical_uncallable(self): 
        with self.assertRaisesRegex(TypeError, 'test must be callable'):
            random.constraints.GenericDynamical(0).satisfies(s_pombe)
    def test_generic_dynamical_graph_type(self): 
        #will return a type error if graph passed in is not a neet network
        def always_true(self):
            return True
        g = nx.DiGraph([(0,1), (1,2), (2,1), (3,2)])
        with self.assertRaisesRegex(TypeError, 'only neet networks are testable with dynamical constraints'):
            random.constraints.GenericDynamical(always_true).satisfies(g)