# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import unittest
from neet.asynchronous import *

class TestAsync(unittest.TestCase):
    def test_canary(self):
        self.assertEqual(3, 1+2)
