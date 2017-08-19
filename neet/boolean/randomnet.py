"""
Functions that generate random networks from a given network.
"""
# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

import random
from neet.statespace import StateSpace
from .logicnetwork import LogicNetwork


def random_logic(logic_net, p=0.5, connections='fixed'):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    `connections` decides how a node in the random network is connected from
    other nodes. With the `'fixed'` option, the random network has the same
    connections as the input network. With the `'shuffled'` option, the number
    of connections to a node is the same as the input network, but the connections
    are randomly selected. With the `'free'` option, the connections in the
    random network do not take reference from the input network, and the number of
    connections to a node and connections themselves are random.

    `p` is the probability of a state of the connected nodes being present in
    the activation table. It is also equavolent to the probability of any node
    being activated. If `p` is a single number, it applies to all nodes. Otherwise
    `p` must be a sequence of numbers that match in size with the input network.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :type p: number or sequence
    :param connections: 'fixed', 'shuffled' or 'free'
    :type connections: str
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    if isinstance(p, (int, float)):
        ps = [p] * logic_net.size
    elif len(p) != logic_net.size:
        raise ValueError("p's length must match with network size")
    else:
        ps = p

    if connections == 'fixed':
        return _random_logic_fixed_connections(logic_net, ps)
    elif connections == 'shuffled':
        return _random_logic_shuffled_connections(logic_net, ps)
    elif connections == 'free':
        return _random_logic_free_connections(logic_net, ps)
    else:
        raise ValueError("connections must be 'fixed', 'shuffled' or 'free'")


def _random_logic_fixed_connections(logic_net, ps):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    Connections in the returned network are the same as those of the input.

    :param logic_net: a :class:LogicNetwork
    :param ps: probability that a state is present in the activation table
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for i, row in enumerate(logic_net.table):
        indices = row[0]

        conditions = set()
        for state in StateSpace(len(indices)):
            if random.random() < ps[i]:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)


def _random_logic_shuffled_connections(logic_net, ps):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    The number of connections to a node is the same as the input network, but
    the connections are randomly selected.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :returns: a random :class:LogicNetwork
    """
    if not isinstance(logic_net, LogicNetwork):
        raise ValueError('object must be a LogicNetwork')

    new_table = []
    for i, row in enumerate(logic_net.table):
        n_indices = len(row[0])
        indices = tuple(sorted(random.sample(
            range(logic_net.size), k=n_indices)))

        conditions = set()
        for state in StateSpace(n_indices):
            if random.random() < ps[i]:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)


def _random_logic_free_connections(logic_net, ps):
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
    for i in range(logic_net.size):
        n_indices = random.randint(1, logic_net.size)
        indices = tuple(sorted(random.sample(
            range(logic_net.size), k=n_indices)))

        conditions = set()
        for state in StateSpace(n_indices):
            if random.random() < ps[i]:
                conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)
