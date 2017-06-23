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
        self._num_states = 2 ** self._size  # To be changed with the StateSpace
        for row in table:
            if not (isinstance(row, (list, tuple)) and len(row) == 2
                    and isinstance(row[0], int) and isinstance(row[1], set)):
                raise ValueError("Invalid table format.")
            if row[0] not in range(self._num_states):
                raise ValueError("mask must be an encoded net state.")
            if any([encode not in range(self._num_states) for encode in row[1]]):
                raise ValueError("active_condition must be encoded net state.")

        self.table = table
        self._state_space = StateSpace(self.size, b=2)

    @property
    def size(self):
        """
        """
        return self._size

    @property
    def state_space(self):
        """
        """
        return self._state_space

    def check_state(self, states):
        """
        """
        if len(states) != self.size:
            raise ValueError("incorrect number of states in array")
        for x in states:
            if x != 0 and x != 1:
                raise ValueError("invalid node state in states")
        return True

    def _update(self, net_state, index=None):
        """
        """
        encoded_state = self.state_space.encode(net_state)

        new_net_state = net_state.copy()

        if index:
            indices = [index]
        else:
            indices = range(self.size)

        for idx in indices:
            mask = self.table[idx][0]
            sub_net_state = mask & encoded_state
            new_net_state[idx] = 1 if sub_net_state in self.table[idx][1] else 0

        return new_net_state

    def update(self, net_state, index=None, pin=None):
        """
        """
        new_net_state = self._update(net_state, index)

        if pin:
            for idx in pin:
                new_net_state[idx] = net_state[idx]

        return new_net_state

    @classmethod
    def read_table(cls, table_file):
        """
        """
        pass
