import unittest
import networkx as nx
import numpy as np
from math import isclose
from neet.boolean import LogicNetwork
from neet.boolean.examples import mouse_cortical_7B, myeloid, s_pombe
from neet.boolean.random import constraints

class TestConstraints(unittest.TestCase):
    def test_has_external_nodes_invalid_type(self):
        with self.assertRaises(TypeError):
            constraints.HasExternalNodes(1.4)
        with self.assertRaises(TypeError):
            constraints.HasExternalNodes('abc')
        with self.assertRaises(TypeError):
            constraints.HasExternalNodes(LogicNetwork([(0,), ['0']]))

    def test_has_external_nodes_integer(self):
        with self.assertRaises(ValueError):
            constraints.HasExternalNodes(-1)

        constraint = constraints.HasExternalNodes(5)
        self.assertEquals(5, constraint.num_external)

        net = nx.DiGraph()
        net.add_nodes_from(range(5))
        self.assertTrue(constraint.satisfies(net))

        net = nx.DiGraph()
        net.add_nodes_from(range(6))
        self.assertFalse(constraint.satisfies(net))

        net = nx.DiGraph()
        net.add_edge(0,1)
        self.assertFalse(constraint.satisfies(net))

    def test_has_external_nodes_graph(self):
        net5 = nx.DiGraph()
        net5.add_nodes_from(range(5))

        net4_01 = nx.DiGraph()
        net4_01.add_nodes_from(range(5))
        net4_01.add_edge(0,1)

        net4_31 = nx.DiGraph()
        net4_31.add_nodes_from(range(5))
        net4_31.add_edge(3,1)

        constraint = constraints.HasExternalNodes(net5)
        self.assertEquals(5, constraint.num_external)
        self.assertTrue(constraint.satisfies(net5))
        self.assertFalse(constraint.satisfies(net4_01))
        self.assertFalse(constraint.satisfies(net4_31))

        constraint = constraints.HasExternalNodes(net4_01)
        self.assertEquals(4, constraint.num_external)
        self.assertTrue(constraint.satisfies(net4_01))
        self.assertTrue(constraint.satisfies(net4_31))
        self.assertFalse(constraint.satisfies(net5))

    def test_is_connected_null_graph(self):
        constraint = constraints.IsConnected()
        net = nx.DiGraph()
        with self.assertRaises(constraints.ConstraintError):
            constraint.satisfies(net)

    def test_is_connected(self):
        constraint = constraints.IsConnected()

        net = nx.DiGraph()
        net.add_node(1)
        self.assertTrue(constraint.satisfies(net))

        net.add_node(2)
        self.assertFalse(constraint.satisfies(net))

        net.add_edge(1,2)
        self.assertTrue(constraint.satisfies(net))

    def test_is_irreducible_invalid_type(self):
        constraint = constraints.IsIrreducible()
        with self.assertRaises(TypeError):
            constraint.satisfies(s_pombe)

    def test_is_irreducible(self):
        constraint = constraints.IsIrreducible()

        net = LogicNetwork([((), ())])
        self.assertTrue(constraint.satisfies(net))

        net = LogicNetwork([((0,), ('0', '1'))])
        self.assertFalse(constraint.satisfies(net))

        net = LogicNetwork([((1,), ('0',)),
                            ((0,), ('1',))])
        self.assertTrue(constraint.satisfies(net))

        net = LogicNetwork([((1,), ('0',)),
                            ((0,), ('0', '1',))])
        self.assertFalse(constraint.satisfies(net))

    def test_has_canalizing_nodes_invalid_type(self):
        with self.assertRaises(TypeError):
            constraints.HasCanalizingNodes(1.5)
        with self.assertRaises(TypeError):
            constraints.HasCanalizingNodes('abc')
        with self.assertRaises(TypeError):
            constraints.HasCanalizingNodes(nx.DiGraph())

    def test_has_canalizing_nodes(self):
        with self.assertRaises(ValueError):
            constraints.HasCanalizingNodes(-1)

        constraint = constraints.HasCanalizingNodes(11)
        self.assertTrue(constraint.satisfies(myeloid))

        constraint = constraints.HasCanalizingNodes(8)
        self.assertFalse(constraint.satisfies(myeloid))

        constraint = constraints.HasCanalizingNodes(13)
        self.assertFalse(constraint.satisfies(myeloid))

        constraint = constraints.HasCanalizingNodes(myeloid)
        self.assertTrue(constraint.satisfies(myeloid))
        self.assertFalse(constraint.satisfies(s_pombe))

        constraint = constraints.HasCanalizingNodes(5)
        self.assertTrue(constraint.satisfies(s_pombe))

        constraint = constraints.HasCanalizingNodes(3)
        self.assertFalse(constraint.satisfies(s_pombe))

        constraint = constraints.HasCanalizingNodes(7)
        self.assertFalse(constraint.satisfies(s_pombe))

        constraint = constraints.HasCanalizingNodes(s_pombe)
        self.assertFalse(constraint.satisfies(myeloid))
        self.assertTrue(constraint.satisfies(s_pombe))

    def test_generic_topological_constraint(self):
        with self.assertRaises(TypeError):
            constraints.GenericTopologicalConstraint(None)
        with self.assertRaises(TypeError):
            constraints.GenericTopologicalConstraint(1)
        with self.assertRaises(TypeError):
            constraints.GenericTopologicalConstraint(1.5)

        constraint = constraints.GenericTopologicalConstraint(lambda g: True)

        with self.assertRaises(TypeError):
            constraint.satisfies(myeloid)

        net = nx.DiGraph()
        self.assertTrue(constraint.satisfies(net))
        net.add_nodes_from(range(5))
        self.assertTrue(constraint.satisfies(net))

        constraint = constraints.GenericTopologicalConstraint(lambda g: len(g) == 5)
        net = nx.DiGraph()
        self.assertFalse(constraint.satisfies(net))
        net.add_nodes_from(range(5))
        self.assertTrue(constraint.satisfies(net))

    def test_generic_dynamical_constraint(self):
        with self.assertRaises(TypeError):
            constraints.GenericDynamicalConstraint(None)
        with self.assertRaises(TypeError):
            constraints.GenericDynamicalConstraint(1)
        with self.assertRaises(TypeError):
            constraints.GenericDynamicalConstraint(1.5)

        constraint = constraints.GenericDynamicalConstraint(lambda g: True)

        with self.assertRaises(TypeError):
            constraint.satisfies(nx.DiGraph())

        self.assertTrue(constraint.satisfies(myeloid))
        self.assertTrue(constraint.satisfies(s_pombe))

        def expect_mean_bias(bias):
            def mean_bias(net):
                if not isinstance(net, LogicNetwork):
                    raise constraints.ConstraintError()
                return np.mean([float(len(row[1])) / float(2**len(row[0]))
                                for row in net.table])
            return lambda net: isclose(mean_bias(net), bias)

        constraint = constraints.GenericDynamicalConstraint(expect_mean_bias(0.2840909090909091))
        self.assertTrue(constraint.satisfies(myeloid))
        self.assertFalse(constraint.satisfies(mouse_cortical_7B))
        with self.assertRaises(constraints.ConstraintError):
            constraint.satisfies(s_pombe)

    def test_generic_node_constraint(self):
        with self.assertRaises(TypeError):
            constraints.GenericNodeConstraint(None)
        with self.assertRaises(TypeError):
            constraints.GenericNodeConstraint(1)
        with self.assertRaises(TypeError):
            constraints.GenericNodeConstraint(1.5)

        constraint = constraints.GenericNodeConstraint(lambda cond: True)

        with self.assertRaises(TypeError):
            constraint.satisfies(myeloid)

        self.assertTrue(constraint.satisfies(set()))
        self.assertTrue(constraint.satisfies(set(['111', '110'])))

        constraint = constraints.GenericNodeConstraint(lambda cond: '111' in cond)
        self.assertTrue(constraint.satisfies(set(['111'])))
        self.assertFalse(constraint.satisfies(set(['110', '000'])))
        self.assertTrue(constraint.satisfies(set(['110', '000', '111'])))

    def test_irreducible_node(self):
        constraint = constraints.IrreducibleNode()
        self.assertTrue(constraint.satisfies(set()))
        self.assertTrue(constraint.satisfies(set(['1'])))
        self.assertFalse(constraint.satisfies(set(['0', '1'])))
        self.assertTrue(constraint.satisfies(set(['00', '11'])))
        self.assertFalse(constraint.satisfies(set(['00', '01'])))

        with self.assertRaises(TypeError):
            constraint.satisfies(myeloid)
