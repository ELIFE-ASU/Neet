# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet

class TestCore(unittest.TestCase):
    class IsNetwork(object):
        def update(self):
            pass

    class IsNotNetwork(object):
        pass

    def test_is_network(self):
        net = self.IsNetwork()
        self.assertTrue(neet.is_network(net))
        self.assertFalse(neet.is_network(self.IsNetwork))
        
        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_network(not_net))
        self.assertFalse(neet.is_network(self.IsNotNetwork))
        
        self.assertFalse(neet.is_network(5))
        self.assertFalse(neet.is_network(int))

    def test_is_network_type(self):
        net = self.IsNetwork()
        self.assertFalse(neet.is_network_type(net))
        self.assertTrue(neet.is_network_type(self.IsNetwork))
        
        not_net = self.IsNotNetwork()
        self.assertFalse(neet.is_network_type(not_net))
        self.assertFalse(neet.is_network_type(self.IsNotNetwork))
           
        self.assertFalse(neet.is_network_type(5))
        self.assertFalse(neet.is_network_type(int))
        