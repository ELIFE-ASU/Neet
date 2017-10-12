# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.boolean.examples as ex

def all_example_networks():
        return [
            ex.s_pombe,
            ex.s_cerevisiae,
            ex.c_elegans,
            ex.p53_no_dmg,
            ex.p53_dmg,
            ex.mouse_cortical_ant_init_7B,
            ex.mouse_cortical_ant_final_7B,
            ex.mouse_cortical_ant_init_7C,
            ex.mouse_cortical_ant_final_7C,
            ex.mouse_cortical_post_init_7B,
            ex.mouse_cortical_post_final_7B,
            ex.mouse_cortical_post_init_7C,
            ex.mouse_cortical_post_final_7C,
            ex.myeloid,
        ]

class TestBooleanExamples(unittest.TestCase):
    
    def test_examples_loaded(self):
        """
        Test that all example networks successfully load.
        """
        all_example_networks()
            
    def test_s_pombe(self):
        self.assertEqual(9, ex.s_pombe.size)
        self.assertEqual(["SK", "Cdc2_Cdc13","Ste9","Rum1","Slp1","Cdc2_Cdc13_active","Wee1_Mik1","Cdc25","PP"],
            ex.s_pombe.names)


    def test_s_cerevisiae(self):
        self.assertEqual(11, ex.s_cerevisiae.size)
        self.assertEqual(["Cln3", "MBF","SBF","Cln1_2","Cdh1","Swi5","Cdc20_Cdc14","Clb5_6","Sic1","Clb1_2","Mcm1_SFF"],
            ex.s_cerevisiae.names)

    def test_mouse_cortical(self):
        mouse_networks = [ex.mouse_cortical_ant_init_7B,
        ex.mouse_cortical_ant_final_7B,
        ex.mouse_cortical_ant_init_7C,
        ex.mouse_cortical_ant_final_7C,
        ex.mouse_cortical_post_init_7B,
        ex.mouse_cortical_post_final_7B,
        ex.mouse_cortical_post_init_7C,
        ex.mouse_cortical_post_final_7C]

        for mouse_network in mouse_networks:
            self.assertEqual(10, mouse_network.size)
            self.assertEqual(["gF","gE","gP","gC","gS","pF","pE","pP","pC","pS"],
                mouse_network.names)

    def test_examples_metadata(self):
        """
        Test that all examples have name, description, and citation metadata.
        """
        for net in all_example_networks():
            self.assertTrue('name' in net.metadata)
            self.assertTrue('description' in net.metadata)
            self.assertTrue('citation' in net.metadata)

