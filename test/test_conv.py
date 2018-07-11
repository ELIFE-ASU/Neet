# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""Unit tests for network conversion."""
import unittest
import neet.boolean.examples as ex
import neet.boolean.conv as conv
from neet.boolean.wtnetwork import WTNetwork
from neet.boolean.logicnetwork import LogicNetwork

class TestConv(unittest.TestCase):
    def test_wt_to_logic(self):
        with self.assertRaises(TypeError):
            conv.wt_to_logic(LogicNetwork([((0,), {'0'})]))

        net = conv.wt_to_logic(ex.s_pombe)
        truth_table = [((0,), set()),
                       ((2, 3, 4), {'000'}),
                       ((0, 1, 2, 5, 8), {'00001', '00100',
                                          '00101', '00111', '01101', '10101'}),
                       ((0, 1, 3, 5, 8), {'00001', '00100',
                                          '00101', '00111', '01101', '10101'}),
                       ((5,), {'1'}),
                       ((2, 3, 4, 6, 7), {'00001'}),
                       ((1, 6, 8), {'001', '010', '011', '111'}),
                       ((1, 7, 8), {'010', '100', '110', '111'}),
                       ((4,), {'1'})]
        self.assertEqual(net.table, truth_table)
