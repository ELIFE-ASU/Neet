# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest

import neet

class TestCanary(unittest.TestCase):
    def test_add_one_two(self):
        self.assertEqual(3, 1+2)
