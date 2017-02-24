# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest

import neet

class CanaryTests(unittest.TestCase):
    def test_addOneTwo(self):
        self.assertEqual(3, 1+2)
