"""
Functions that generate random networks from a given network.
"""
# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

import random
import numpy as np
from neet.statespace import StateSpace
from .logicnetwork import LogicNetwork



def random_logic(logic_net, p=0.5, connections='fixed-structure'):
    """
    Return a `LogicNetwork` from an input `LogicNetwork` with a random logic table.

    `connections` decides how a node in the random network is connected from
    other nodes. With the `'fixed-structure'` option, the random network has the same
    connections as the input network. With the `'fixed-in-degree'` option, the number
    of connections to a node is the same as the input network, but the connections
    are randomly selected. With the 'fixed-mean-degree' option, the total number of
    edges is conserved, but edges are placed randomly between nodes.  With the 
    `'free'` option, only the number of nodes is conserved, with the number of 
    connections to a node chosen uniformly between 1 and the total number of nodes.

    `p` is the probability of a state of the connected nodes being present in
    the activation table. It is also equavolent to the probability of any node
    being activated. If `p` is a single number, it applies to all nodes. Otherwise
    `p` must be a sequence of numbers that match in size with the input network.

    :param logic_net: a :class:LogicNetwork
    :param p: probability that a state is present in the activation table
    :type p: number or sequence
    :param connections: 'fixed-structure', 'fixed-in-degree', 'fixed-mean-degree', or 'free'
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

    if connections == 'fixed-structure':
        return _random_logic_fixed_connections(logic_net, ps)
    elif connections == 'fixed-in-degree':
        return _random_logic_shuffled_connections(logic_net, ps)
    elif connections == 'fixed-mean-degree':
        return _random_logic_fixed_num_edges(logic_net, ps)
    elif connections == 'free':
        return _random_logic_free_connections(logic_net, ps)
    else:
        raise ValueError("connections must be 'fixed', 'fixed-in-degree', 'fixed-mean-degree', or 'free'")


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


# 10.26.2017
def _random_logic_fixed_num_edges(logic_net,ps):
    """
    Returns new network that corresponds to adding a fixed number of
    edges between random nodes, with random corresponding boolean rules.
    """
    
    numEdges = np.sum(_degrees(logic_net))
    # choose n random integers that sum to numEdges
    newDegrees = _random_partition(logic_net.size,numEdges,m=logic_net.size)
    
    new_table = []
    for i,degree in enumerate(newDegrees):
        n_indices = degree
        indices = tuple(sorted(random.sample(
            range(logic_net.size), k=n_indices)))

        conditions = set()
        
        if n_indices > 0:
            for state in StateSpace(n_indices):
                if random.random() < ps[i]:
                    conditions.add(tuple(state))

        new_table.append((indices, conditions))

    return LogicNetwork(new_table, logic_net.names)


def _degrees(net):
    """
    Return the list of node in-degrees for the network.
    """
    return [ len(t[0]) for t in net.table ]


def _random_partition(n,s,m=np.inf):
    """
    Choose n random integers that sum to s, with the maximum value
    of any element of the list limited to m.
    """
    if s > n*m:
        raise Exception, "Can't have s > n*m"
    
    # see, e.g., https://stackoverflow.com/questions/5622608/choosing-n-numbers-with-fixed-sum
    partition = [0] + list(np.random.randint(0,s+1,n-1)) + [s]
    partition = np.sort(partition)
    integers = partition[1:] - partition[:-1]
    
    # redistribute any values above the max
    # (there's probably a better way to do this!)
    while max(integers) > m:
        maxedIndices = ( integers >= m )
        nonMaxedIndices = ( integers < m )
        numToRedistribute = np.sum(integers[maxedIndices] - m)
        redistributed = _random_partition(sum(nonMaxedIndices),numToRedistribute)
        integers[maxedIndices] = m
        integers[nonMaxedIndices] += redistributed
        
    return integers

