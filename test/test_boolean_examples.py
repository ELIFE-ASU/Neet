# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
import neet.boolean.examples as ex

class TestBooleanExamples(unittest.TestCase):
    def test_s_pombe(self):
        self.assertEqual(9, ex.s_pombe.size)
        self.assertEqual(["SK", "Cdc2_Cdc13","Ste9","Rum1","Slp1","Cdc2_Cdc13_active","Wee1_Mik1","Cdc25","PP"],
            ex.s_pombe.names)


    def test_s_cerevisiae(self):
        self.assertEqual(11, ex.s_cerevisiae.size)
        self.assertEqual(["Cln3", "MBF","SBF","Cln1_2","Cdh1","Swi5","Cdc20_Cdc14","Clb5_6","Sic1","Clb1_2","Mcm1_SFF"],
            ex.s_cerevisiae.names)
