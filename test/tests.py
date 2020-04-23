from .mock import MockObject, MockNetwork, MockUniformNetwork
from neet import Network, UniformNetwork
from neet.boolean import logicnetwork, network
from neet.boolean.random import dynamics
from neet.boolean.examples import s_pombe
import math
import unittest


class TestNetwork(unittest.TestCase):

    net1 = logicnetwork.LogicNetwork([((1,2,3), {'000', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])
    net2 = logicnetwork.LogicNetwork([((0,), {'0', '1'}), ((1,2,3), {'000', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])
    net3 = logicnetwork.LogicNetwork([((1,2,3), {'000', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])
    net4 = logicnetwork.LogicNetwork([((1,2,3), {'001', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])

    netList = [net1, net2, net3, net4]

    #errorNet1 = logicnetwork.LogicNetwork([])

    def test_uniform_bias(self, net=None, printq=True, debug=False):
        if (net == None):
            for network in self.netList:
                self.test_uniform_bias(network, printq=False)
            print("test passed!")
            return
        r = dynamics.UniformBias(net, p=0.625)
        randomNet = r.random()

        if(debug):
            print(net.table)
            print(randomNet.table)

        if(printq):
            print("test passed!")

    def test_external_exclusion(self, net=None, printq=True, debug=False):
        # Node 0 is an external node
        #net = logicnetwork.LogicNetwork([((0,), {'0', '1'}), ((1,2,3), {'000', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])
        if (net == None):
            for network in self.netList:
                self.test_external_exclusion(network, False)
            print("test passed!")
            return

        r = dynamics.UniformBias(net, p=0.625)
        randomNet = r.random()
        net.reduce_table()
        x = 0
        for row in net.table:
            if( row[0] == tuple([x]) ):#if the node depends upon itself (is external)
                if( not row[1] == randomNet.table[x][1] ):#compare to the new network's node which should still/also be external
                    print("the dynamic randomizer has changed an external node")
                    print(row[1])
                    print(net.table[x][1])
                    raise ValueError("the dynamic randomizer has changed an external node")
            x += 1
                    
        if(debug):
            print(net.table)
            print(randomNet.table)

        if(printq):
            print("test passed!")
    
    def test_correct_bias(self, net=None, printq=True, debug=False):
        #net = logicnetwork.LogicNetwork([((1,2,3), {'000', '100'}), ((1,2,3), {'101', '011'}), ((1,2,3), {'111'}), ((1,2,3), {'000', '111', '010'})])
        if (net == None):
            for network in self.netList:
                self.test_correct_bias(network, False)
            print("test passed!")
            return
        #print(net.table)
        p = 0.1
        r = dynamics.UniformBias(net, p)
        randomNet = r.random()
        net.reduce_table()
        x = 0
        for row in randomNet.table:
            if( not ( row[0] == tuple([x]) ) ):
                numPossibleInputs = math.pow(2.0, float(len(row[0])))
                upperBound = math.ceil(p * numPossibleInputs)
                lowerBound = math.floor(p * numPossibleInputs)
                if not (len(row[1]) == upperBound or len(row[1]) == lowerBound):
                    raise Exception("Randomizer failed to create an appropriate Bias")
            x += 1

        if(debug):
            print(net.table)
            print(randomNet.table)

        if(printq):
            print("test passed!")

    def test_correct_mean(self, net=None, printq=True, debug=False):
        if (net == None):
            for network in self.netList:
                self.test_correct_mean(network, printq=False)
            print("test passed!")
            return
        r = dynamics.MeanBias(net)
        randomNet = r.random()

        bias_sum_one = 0.0
        row_count_one = 0.0
        bias_sum_two = 0.0
        row_count_two = 0.0

        if(debug):
            print(type(bias_sum_one))
        for row in net.table:
            bias_sum_one += float(len(row[1]))
            row_count_one += 1.0
        mean_one = bias_sum_one / row_count_one
        for row in randomNet.table:
            bias_sum_two += float(len(row[1]))
            row_count_two += 1.0
        mean_two = bias_sum_two / row_count_two

        
        if not (mean_one == mean_two):
            raise ValueError("the mean bias of the two networks is not the same")
        if not (row_count_one == row_count_two):
            raise ValueError("the networks created do not have the same topology")

        if(debug):
            print(net.table)
            print("bias_1: ", bias_sum_one)
            print("row_count_1: ", row_count_one)
            print("mean_1: ", bias_sum_one / row_count_one)
            print(randomNet.table)
            print("bias_2: ", bias_sum_two)
            print("row_count_2: ", row_count_two)
            print("mean_2: ", bias_sum_two / row_count_two)

        if(printq):
            print("test passed!")

    def test_correct_errors_thrown(self):
        with self.assertRaises(NotImplementedError):
            r = dynamics.MeanBias(s_pombe)
        with self.assertRaises(NotImplementedError):
            r = dynamics.LocalBias(s_pombe)
        with self.assertRaises(NotImplementedError):
            
            r = dynamics.MeanBias(s_pombe)
        print("test passed!")
        
