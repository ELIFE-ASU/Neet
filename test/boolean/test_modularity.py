import unittest
import modularity as md

from neet.boolean.examples import s_pombe, s_cerevisiae
from neet.boolean import LogicNetwork

TEST_CELL_COLLECTIVE = True

if TEST_CELL_COLLECTIVE:
    import datadex
    from load_cell_collective_nets import loadCellCollectiveNets

def atts_and_cks_modular_and_sampled(net,numsamples=100,seed=123,iterative=False):
    """
    Calculate attractors and control kernels using both exact "modular" approach
    and "sampled" appraoch.
    
    Returns: atts, cks, sampled_atts, sampled_cks
    """
    # calculate using modular code
    atts,outdict = md.attractors(net,find_control_kernel=True)
    cks = outdict['control_kernels']
    
    # calculate using sampling code
    sampled_atts = md.sampled_attractors(net,numsamples=numsamples,seed=seed)
    sampled_cks = md.sampled_control_kernel(net,numsamples=numsamples,
                                            phenotype='all',seed=seed,
                                            iterative=iterative,
                                            iterative_rounds_out=False)
                                            
    return atts,cks,sampled_atts,sampled_cks

class TestModularityHelpers(unittest.TestCase):
    
    def test_attractors_equivalent(self):
        a1 = [1,2,3,4]
        a2 = [2,3,4,1]
        a3 = [4,1,2,3]
        a4 = [3,2,1,4]
        a5 = [1,2,3,8]
        a6 = [1,2,3]
        self.assertTrue(md.attractors_equivalent(a1,a1))
        self.assertTrue(md.attractors_equivalent(a1,a2))
        self.assertTrue(md.attractors_equivalent(a1,a3))
        self.assertTrue(md.attractors_equivalent(a2,a3))
        self.assertFalse(md.attractors_equivalent(a1,a4))
        self.assertFalse(md.attractors_equivalent(a1,a5))
        self.assertFalse(md.attractors_equivalent(a1,a6))

    def test_atts_and_cks_equivalent(self):
        atts1 = [[1,2,3,4],[5,6,7],[8,]]
        cks1 = [{0,1,2},{0,},None]
        atts2 = [[6,7,5],[3,4,1,2],[8,]]
        cks2 = [{0,},{1,0,2},None]
        atts3 = [[6,7,5],[3,4,1,2],[8,]]
        cks3 = [{1,0,2},{0,},None]
        atts4 = [[1,2,3,4],[5,6,7]]
        cks4 = [{0,1,2},{0,}]
        atts5 = [[1,2,3,4],[8,],[5,6,7]]
        cks5 = [{10,11,12},None,{13,}]
        atts6 = [[1,2,3,4],[8,],[5,6,7]]
        cks6 = [{10,11,12},{0,},{13,}]
        self.assertTrue(md.atts_and_cks_equivalent(atts1,cks1,atts1,cks1))
        self.assertTrue(md.atts_and_cks_equivalent(atts1,cks1,atts2,cks2))
        self.assertFalse(md.atts_and_cks_equivalent(atts1,cks1,atts3,cks3))
        self.assertFalse(md.atts_and_cks_equivalent(atts1,cks1,atts4,cks4))
        self.assertFalse(md.atts_and_cks_equivalent(atts1,cks1,atts5,cks5))
        self.assertTrue(md.atts_and_cks_equivalent(atts1,cks1,atts5,cks5,
                                                   ck_size_only=True))
        self.assertFalse(md.atts_and_cks_equivalent(atts1,cks1,atts6,cks6,
                                                   ck_size_only=True))
                                                   
    def test_distinguishing_nodes(self):
        attractors = [[[1,0,0]],
                      [[0,1,0]],
                      [[0,0,1]],
                      [[0,0,0]]]
        correct_dn_list = [ [{0,},{0,1},{0,2},{0,1,2}],
                            [{1,},{0,1},{1,2},{0,1,2}],
                            [{2,},{0,2},{1,2},{0,1,2}],
                            [{0,1,2}] ]
        dn_gen_list = md.distinguishing_nodes_from_attractors(attractors)
        self.assertTrue(len(dn_gen_list) == len(attractors))
        for dn_gen,correct_dn in zip(dn_gen_list,correct_dn_list):
            dn = [ d for d in dn_gen ]
            self.assertTrue(len(dn) == len(correct_dn))
            for d in dn:
                self.assertTrue(d in correct_dn)
        

class TestModularity(unittest.TestCase):
    
    def test_attractors_s_cerevisiae(self):
        atts = md.attractors(s_cerevisiae)
        known_fixed_points = [0,272,12,16,256,274,258] # depends on neet encoding...
        self.assertEqual(len(known_fixed_points),len(atts))
        for fp in known_fixed_points:
            self.assertTrue(md.np.array([fp]) in atts)
    
    def test_control_kernel_s_pombe(self):
        test_attractor = md.np.array([76]) # note this depends on neet encoding...
        atts,outdict = md.attractors(s_pombe,find_control_kernel=True)
        test_index = atts.index(test_attractor)
        self.assertEqual({2,3,6,7},outdict['control_kernels'][test_index])

    def test_control_kernel_s_cerevisiae(self):
        test_attractor = md.np.array([272]) # note this depends on neet encoding...
        atts,outdict = md.attractors(s_cerevisiae,find_control_kernel=True)
        test_index = atts.index(test_attractor)
        self.assertEqual({1,2,4,8},outdict['control_kernels'][test_index])

    def test_control_kernel_cycle(self):
        table = [((0,),{'1',}),
                 ((0,1,),{'01','10'})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [0,], [2,], [1,3] ]
        correct_cks = [ {0,1}, {0,1}, {0,} ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks))
                                                   
    def test_control_simple_switcher(self):
        table = [((0,),{'0',})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [0,1], ]
        correct_cks = [ set(), ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks))
                                                   
    def test_control_simple_uncontrollable_cycle(self):
        table = [((1,),{'0',}),
                 ((0,),{'0',})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [1,], [2,], [0,3] ]
        correct_cks = [ {0,}, {0,}, None ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks,
                                                   ck_size_only=True))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks,
                                                   ck_size_only=True))
                                                   
    def test_control_simple_asynchronous_switcher(self):
        table = [((0,),{'0',}),
                 ((1,),{'0',})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [1,2], [0,3] ]
        correct_cks = [ None, None ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks))
        
    def test_control_tricky_1(self):
        """
        "Counterexample" case from notes 4/22/2020.  For this case,
        the iterative and non-iterative methods give different results,
        because the smallest distinguishing node set is not in the
        final control kernel for the 0 attractor.
        """
        table = [((0,1,2),{'001','010','011','101','110'}),
                 ((0,1,2),{'010','011','110'}),
                 ((0,1,2),{'001','011','101'})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [0], [3], [5] ]
        correct_cks = [ {1,2}, {1}, {2} ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks))
                                                   
        # also check that sampled_control_kernel with iterative=True
        # fails in the expected way
        expected_cks_iterative = [ {0,1,2}, {1}, {2} ]
        _,_,sampled_atts_iterative,sampled_cks_iterative = \
                            atts_and_cks_modular_and_sampled(net,iterative=True)
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,expected_cks_iterative,
                                                   sampled_atts_iterative,
                                                   sampled_cks_iterative))
                                          
        # also check the "iterative rounds" functionality
        _,iterative_rounds = md.sampled_control_kernel(net,numsamples=100,
                                                phenotype='all',seed=123,
                                                iterative=True,
                                                iterative_rounds_out=True)
        self.assertTrue([4,1] in iterative_rounds)
        self.assertTrue([1] in iterative_rounds)
        
    
    def test_control_tricky_2(self):
        """
        Example case from notes 5/8/2020.  For this case, the control
        kernel includes nodes that have constant values over the
        original attractors.
        """
        table = [((0,1,2),{'001','010','011','100'}),
                 ((0,1),{'01'}),
                 ((0,2),{'01'})]
        net = LogicNetwork(table)
        
        atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
        
        # note that encoded attractors depend on the neet encoding
        correct_atts = [ [0], [1] ]
        correct_cks = [ {0,1,2}, {0} ]
        
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   atts,cks))
        self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                   sampled_atts,sampled_cks))
    
    def test_control_iron(self):
        if TEST_CELL_COLLECTIVE:
            cc_iron_index = 37
            cc_iron_name = 'Iron Acquisition And Oxidative Stress Response In Aspergillus Fumigatus.'
            net = loadCellCollectiveNets(cc_iron_index)[cc_iron_name]
        
            atts,cks,sampled_atts,sampled_cks = atts_and_cks_modular_and_sampled(net)
            
            # note that encoded attractors depend on the neet encoding
            correct_atts = [[462114, 470306, 486435, 97959, 581343,
                             562060, 920586, 69030, 396546],
                            [3084075, 3092267, 3108395, 2752175, 2678495,
                             2659212, 3017738, 2166183, 2494211],
                            [1684397, 1986139],
                            [3781549, 4083291]]
            correct_cks = [{20, 21}, {20, 21}, {20, 21}, {20, 21}]
            
            self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                    atts,cks))
            self.assertTrue(md.atts_and_cks_equivalent(correct_atts,correct_cks,
                                                    sampled_atts,sampled_cks))
                                                    
        
if __name__ == "__main__":
    unittest.main()
