import unittest
import networkx as nx
import numpy as np
from itertools import islice
from neet.boolean.random import topology
from neet.boolean.random.constraints import ConstraintError, HasExternalNodes


class TestTopology(unittest.TestCase):
    def test_fixed_topology_without_constraints(self):
        for net in self.erdos_renyi(10, 10, 0.5, directed=True):
            gen = topology.FixedTopology(net)
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

    def test_fixed_topology_with_init_constraints(self):
        for net in self.erdos_renyi(10, 10, 0.5, directed=True):
            num_external = np.count_nonzero([d == 0 for _, d in net.in_degree()])
            gen = topology.FixedTopology(net, constraints=[HasExternalNodes(num_external)])
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            with self.assertRaises(ConstraintError):
                topology.FixedTopology(net, constraints=[HasExternalNodes(num_external + 1)])

            gen = topology.FixedTopology(net, constraints=[lambda net: len(net) == 10])
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            with self.assertRaises(ConstraintError):
                topology.FixedTopology(net, constraints=[lambda net: len(net) != 10])

    def test_fixed_topology_with_added_constraints(self):
        for net in self.erdos_renyi(10, 10, 0.5, directed=True):
            num_external = np.count_nonzero([d == 0 for _, d in net.in_degree()])
            gen = topology.FixedTopology(net)
            gen.add_constraint(HasExternalNodes(num_external))
            self.assertEquals(1, len(gen.constraints))
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            with self.assertRaises(ConstraintError):
                gen.add_constraint(HasExternalNodes(num_external + 1))
            self.assertEquals(1, len(gen.constraints))

            gen.add_constraint(lambda net: len(net) == 10)
            self.assertEquals(2, len(gen.constraints))
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            with self.assertRaises(ConstraintError):
                gen.add_constraint(lambda net: len(net) != 10)
            self.assertEquals(2, len(gen.constraints))

            with self.assertRaises(TypeError):
                gen.add_constraint(None)
            with self.assertRaises(TypeError):
                gen.add_constraint(1)
            with self.assertRaises(TypeError):
                gen.add_constraint(1.5)
            with self.assertRaises(TypeError):
                gen.add_constraint(True)

    def test_fixed_topology_with_set_constraints(self):
        for net in self.erdos_renyi(10, 10, 0.5, directed=True):
            num_external = np.count_nonzero([d == 0 for _, d in net.in_degree()])
            gen = topology.FixedTopology(net)
            gen.constraints = [HasExternalNodes(num_external)]
            self.assertEquals(1, len(gen.constraints))
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            with self.assertRaises(ConstraintError):
                gen.constraints = [HasExternalNodes(num_external + 1)]
            self.assertEquals(1, len(gen.constraints))

            gen.constraints = [HasExternalNodes(num_external), lambda net: len(net) == 10]
            self.assertEquals(2, len(gen.constraints))
            for r in islice(gen, 10):
                self.assertTrue(nx.is_isomorphic(net, r))

            gen.constraints = []
            self.assertEquals(0, len(gen.constraints))
            with self.assertRaises(ConstraintError):
                gen.constraints = [HasExternalNodes(num_external + 1), lambda net: len(net) != 10]
            self.assertEquals(0, len(gen.constraints))

            with self.assertRaises(TypeError):
                gen.constraints = [None]
            with self.assertRaises(TypeError):
                gen.constraints = [1]
            with self.assertRaises(TypeError):
                gen.constraints = [1.5]
            with self.assertRaises(TypeError):
                gen.constraints = [True]

            with self.assertRaises(TypeError):
                gen.constraints = HasExternalNodes(num_external)

            with self.assertRaises(TypeError):
                gen.constraints = lambda net: len(net) == 10

    #  def test_mean_degree_without_constraints(self):
    #      for net in self.erdos_renyi(10, 10, 0.5, directed=True):
    #          mean_degree = mean
    #          gen = topology.MeanDegree(net)
    #          for r in islice(gen, 10):
    #              neg.degree
    #              self.assertTrue(nx.is_isomorphic(net, r))

    def erdos_renyi(self, num, *args, **kwargs):
        return [nx.generators.random_graphs.erdos_renyi_graph(*args, **kwargs) for _ in range(num)]
