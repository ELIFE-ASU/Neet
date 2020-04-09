import unittest
from neet.boolean.examples import myeloid, s_pombe
from neet.boolean import random

class TestConstraints(unittest.TestCase):
    #Topological constraints test cases
    #HasExternalNodes test cases
    def test_negative_external_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasExternalNodes, -1)
    def test_int_external_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasExternalNodes, 0.6)
    def test_has_external_nodes(self):
        #dummy code because I still need to implement this
        x = 3
    #IsConnected test cases
    #GenericTopological test cases

    #--------------------------------
    #Dynamical Constraints Test Cases:
    #IsIrreducible test cases
    #HasCanalizingNodes test cases
    def test_negative_canalizing_nodes(self):
        #check that you hit a value error if a non-negative target is passed in
        self.assertRaises(ValueError, random.constraints.HasCanalizingNodes, -1)
    def test_int_canalizing_nodes(self):
        #check that you hit a type error if a non-int is passed in
        self.assertRaises(TypeError, random.constraints.HasCanalizingNodes, 0.6)
    #GenericDynamical test cases