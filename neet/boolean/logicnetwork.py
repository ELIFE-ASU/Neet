# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
from neet.statespace import StateSpace


class LogicNetwork(object):
    """
    """

    def __init__(self, table):
        """
        """
        if not isinstance(table, (list, tuple)):
            raise TypeError("table must be a list or tuple.")
        self._size = len(table)
        self._num_states = 2 ** self._size
        for row in table:
            if not (isinstance(row, (list, tuple)) and len(row) == 2
                    and isinstance(row[0], int) and isinstance(row[1], set)):
                raise ValueError("Invalid table format.")
            if row[0] not in range(self._num_states):
                raise ValueError("mask must be an encoded net state.")
            if any([encode in range(self._num_states) for encode in row[1]]):
                raise ValueError("active_condition must be encoded net state.")

        self.table = table

    @property
    def size(self):
        """
        """
        return self._size

    def state_space(self):
        """
        """
        return StateSpace(self.size, b=2)

    def check_state(self, states):
        """
        """
        if len(states) != self.size:
            raise ValueError("incorrect number of states in array")
        for x in states:
            if x != 0 and x != 1:
                raise ValueError("invalid node state in states")
        return True

    def update(self, states, index=None, pin=None):
        """
        """
        pass

    @classmethod
    def read_table(cls, table_file):
        """
        """
        pass
