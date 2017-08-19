"""
Functions that generate random networks from a given network.
"""
# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

import random
from neet.statespace import StateSpace
from .logicnetwork import LogicNetwork


def random_logic(logic_net, p=0.5, fixed_connections=True):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :param fixed_connections: whether the returned network has the same connections as input
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    if fixed_connections:
        return random_logic_fixed_connections(logic_net, p)
    else:
        return random_logic_free_connections(logic_net, p)


def random_logic_fixed_connections(logic_net, p=0.5):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    Connections in the returned network are the same as those of the input.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for row in logic_net.table:
        indices = row[0]

        conditions = set()
        for state in StateSpace(len(indices)):
            if random.random() < p:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)


def random_logic_free_connections(logic_net, p=0.5):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    All possible connections within the network are considered in the random process.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for _ in range(logic_net.size):
        n_indices = random.randint(1, logic_net.size)
        indices = tuple(sorted(random.sample(
            range(logic_net.size), k=n_indices)))

        conditions = set()
        for state in StateSpace(n_indices):
            if random.random() < p:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)
