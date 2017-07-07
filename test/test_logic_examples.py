# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
"""Unit test for LogicNetwork examples"""
import unittest
import neet.boolean.examples as ex


class TestLogicExamples(unittest.TestCase):
    def test_myeloid(self):
        self.assertEqual(11, ex.myeloid.size)
        self.assertEqual(['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL',
                          'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1'], ex.myeloid.names)
        self.assertEqual(ex.myeloid.update([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        # Assert attractors
        self.assertEqual(ex.myeloid.update([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]),
                         [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]),
                         [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]),
                         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
        self.assertEqual(ex.myeloid.update([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
