# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
from neet.statespace import StateSpace


class LogicNetwork(object):
    """
    The LogicNetwork class represents boolean networks whose update rules
    follow logic relations among nodes. Each node state is expressed as ``0``
    or ``1``.
    """

    def __init__(self, table):
        """
        Construct a network from a logic truth table.

        A truth table stores a list of of tuples, one for each node in order.
        The tuple with the form of `(A, {C1, C2, ...})` at index `i` contains
        the activation conditions for the node of index `i`. `A` is a tuple
        marking the indices of the nodes which influence the state of node `i`
        via logic relations. `{C1, C2, ...}` being a set, each element is the
        collective binary state of these influencing nodes that would activate
        node `i`, setting it `1`. Any other collective states of nodes `A` not
        in the set are assumed to deactivate node `i`, setting it `0`. `C1`,
        `C2`, etc. are sequences (`tuple` or `str`) of binary digits, each
        being the binary state of corresponding node in `A`.

        :param table: the logic table

        .. rubric:: Examples

        ::
            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net.size
            1
            >>> net.table
            [((0,), {'0'})]

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net.size
            2
            >>> net.table
            [((1,), {'0', '1'}), ((0,), {'1'})]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), {(0, 1), '10', [1, 1]}),
                                    ((0, 1), {'11'})])
            >>> net.size
            3
            >>> net.table
            [((1, 2), {'01', '10'}), ((0, 2), {'01', '10', '11'}), ((0, 1), {'11'})]

        """
        if not isinstance(table, (list, tuple)):
            raise TypeError("table must be a list or tuple.")

        self.size = len(table)
        self.state_space = StateSpace(self.size, base=2)
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
                    encoded_condition += 2 ** idx if int(state) else 0
                encoded_sub_table.add(encoded_condition)
            self._encoded_table.append((mask_code, encoded_sub_table))
        # Store positive truth table for human reader.
        self.table = table

    def _update(self, net_state, index=None):
        """
        Update node states according to the truth table. Core update function.

        If `index` is provided, update only node at `index`. If `index` is not
        provided, update all ndoes. `pin` provides the indices of which the
        nodes' states are forced to remain unchanged.

        :param net_state: a sequence of binary node states
        :type net_state: sequence
        :param index: the index to update (or None)
        :type index: int or None
        :param pin: the indices to pin (or None)
        :type pin: sequence
        :returns: the updated states

        .. rubric:: Basic Use:

        ::

            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net._update([0], 0)
            [1]
            >>> net._update([1])
            [0]

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net._update([1, 0], 0))
            [1, 0]
            >>> net._update([1, 0], 1))
            [1, 1]
            >>> net._update([0, 0])
            [1, 0]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), {(0, 1), '10', (1, 1)}),
                                    ((0, 1), {'11'})])
            >>> net.size
            3
            >>> net._update([0, 1, 0])
            [1, 0, 0]
            >>> net._update([0, 0, 1])
            [1, 1, 0]
            >>> net._update([0, 0, 1], 1)
            [0, 1, 1]
        """
        encoded_state = self.state_space.encode(net_state)

        new_net_state = net_state[:]  # Python 2.7

        if index is None:
            indices = range(self.size)
        else:
            indices = [index]

        for idx in indices:
            mask, condition = self._encoded_table[idx]
            sub_net_state = mask & encoded_state
            new_net_state[idx] = 1 if sub_net_state in condition else 0

        return new_net_state

    def update(self, net_state, index=None, pin=None):
        """
        Update node states according to the truth table.

        If `index` is provided, update only node at `index`. If `index` is not
        provided, update all ndoes. `pin` provides the indices of which the
        nodes' states are forced to remain unchanged.

        :param net_state: a sequence of binary node states
        :type net_state: sequence
        :param index: the index to update (or None)
        :type index: int or None
        :param pin: the indices to pin (or None)
        :type pin: sequence
        :returns: the updated states

        .. rubric:: Basic Use:

        ::

            >>> net = LogicNetwork([((0,), {'0'})])
            >>> net.update([0], 0)
            [1]
            >>> net.update([1])
            [0]
            >>>

        ::

            >>> net = LogicNetwork([((1,), {'0', '1'}), ((0,), {'1'})])
            >>> net.update([1, 0], 0))
            [1, 0]
            >>> net.update([1, 0], 1))
            [1, 1]
            >>> net.update([0, 0])
            [1, 0]

        ::

            >>> net = LogicNetwork([((1, 2), {'01', '10'}),
                                    ((0, 2), {(0, 1), '10', (1, 1)}),
                                    ((0, 1), {'11'})])
            >>> net.size
            3
            >>> net.update([0, 0, 1], 1)
            [0, 1, 1]
            >>> net.update([0, 1, 0])
            [1, 0, 0]
            >>> net.update([0, 0, 1])
            [1, 1, 0]
            >>> net.update([0, 0, 1], pin=[1])
            [1, 0, 0]
            >>> net.update([0, 0, 1], pin=[0, 1])
            [0, 0, 0]
        """
        if net_state not in self.state_space:
            raise ValueError("the provided state is not in the network's state space")

        new_net_state = self._update(net_state, index)

        if pin:
            for idx in pin:
                new_net_state[idx] = net_state[idx]

        return new_net_state

    @classmethod
    def read_table(cls, table_file):
        """
        """
        # read table from table_file
        # return cls.(table)
        pass
