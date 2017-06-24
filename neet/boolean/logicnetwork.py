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

        self.size = len(table)
        self.state_space = StateSpace(self.size, b=2)
        self._encoded_table = []

        for row in table:
            # Validate mask.
            if not (isinstance(row, (list, tuple)) and len(row) == 2):
                raise ValueError("Invalid table format.")
            # Encode the mask.
            mask_code = 0
            for idx in row[0]:
                if idx >= self.size:
                    raise IndexError("mask index out of range.")
                mask_code += 2 ** idx  # Low order, low index.
            # Validate truth table of the sub net.
            if not isinstance(row[1], (list, tuple, set)):
                raise ValueError("Invalid table format.")
            # Encode each condition of truth table.
            encoded_sub_table = set()
            for condition in row[1]:
                encoded_condition = 0
                for idx, state in zip(row[0], condition):
                    encoded_condition += 2 ** idx if state else 0
                encoded_sub_table.add(encoded_condition)
            self._encoded_table.append((mask_code, encoded_sub_table))
        # Store positive truth table for human reader.
        self.table = table

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
