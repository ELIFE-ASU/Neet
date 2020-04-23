from neet.boolean import random
import networkx as nx
import unittest

class TestTopological(unittest.TestCase):
    #TopologyRandomizer tests
    def test_constraints_is_none(self):
        #check that constraints = [] if no constraints are passed in
        self.assertEqual(random.topology.TopologyRandomizer.constraints(), [])

    #def test_is_topological_constraint_pass(self):
        #check that constraint passes if it is of type TopologicalConstraint
        
    def test_abstract_constraints(self):
        #Check that TypeError is raised if constraint is not callable or type topological
        self.assertRaises(TypeError, random.topology.TopologyRandomizer, """non-abstract constraint""")

    #def test_add_constraint_is_topological_pass(self):
        #Check that new constraint added passes if it is of type topological
       
    def test_add_constraint_abstract_constraints(self):
        #Check that TypeError is raised if newly added constraint is not callable or type Topological   
        self.assertRaises(TypeError, random.topology.TopologyRandomizer, """non-abstract constraint""")


    #FixedTopology tests
    def test_constraints_is_none(self):
        #check that constraints = [] if no constraints are passed in
        self.assertEqual(random.topology.FixedTopology.constraints(), [])

    #def test_is_topological_constraint_pass(self):
        #check that constraint passes if it is of type TopologicalConstraint
        
    def test_abstract_constraints(self):
        #Check that TypeError is raised if constraint is not callable or type topological
        self.assertRaises(TypeError, random.topology.FixedTopology, """non-abstract constraint""")

    def test_constraint_satisfies(self):
        #Check that constraint satisfies self.graph
        self.

    #def test_add_constraint_is_topological_pass(self):
        #Check that new constraint added passes if it is of type topological
        
    def test_add_constraint_abstract_constraints(self):
        #Check that TypeError is raised if newly added constraint is not callable or type Topological
        self.assertRaises(TypeError, random.topology.FixedTopology, """non-abstract constraint""")

    def test_constraint_satisfies(self):
        #Check that constraint satisfies self.graph
        self.

    #MeanDegree tests
    

    #In-Degree tests

    #Out-degree tests



