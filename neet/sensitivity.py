# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_boolean_network
import numpy as np


def sensitivity(net, state):
    """
    Calculate Boolean network sensitivity, as defined in, e.g.,

        Shmulevich, I., & Kauffman, S. A. (2004). Activities and
        sensitivities in Boolean network models. Physical Review 
        Letters, 93(4), 48701.
        http://doi.org/10.1103/PhysRevLett.93.048701

    The sensitivity of a Boolean function f on state vector x is the number of Hamming neighbors of x on which the function value is different than on x.

    .. rubric:: Examples

    ::

        >>> from neet.boolean.examples import s_pombe
        >>> sensitivity(s_pombe,[0,0,0,0,0,1,1,1,1])
        7

    net    : NEET boolean network
    state  : A single network state, represented as a list of node states
    """

    if not is_boolean_network(net):
        raise(TypeError("net must be a boolean network"))

    # list Hamming neighbors
    neighbors = hamming_neighbors(state)

    nextState = net.update(state)

    # count number of neighbors that update to a different state than original
    s = 0
    for neighbor in neighbors:
        newState = net.update(neighbor)
        if not np.array_equal(newState, nextState):
            s += 1

    return s


def hamming_neighbors(state):
    """
    Return Hamming neighbors of a boolean state.

    .. rubric:: Examples

    ::

        >>> hamming_neighbors([0,0,1])
        array([[1, 0, 1],
               [0, 1, 1],
               [0, 0, 0]])
    """
    state = np.asarray(state, dtype=int)
    if len(state.shape) > 1:
        raise(ValueError("state must be 1-dimensional"))
    if not np.array_equal(state % 2, state):
        raise(ValueError("state must be binary"))

    repeat = np.tile(state, (len(state), 1))
    neighbors = (repeat + np.diag(np.ones_like(state))) % 2

    return neighbors


def average_sensitivity(net, states=None, weights=None):
    """
    Calculate average Boolean network sensitivity, as defined in, e.g.,

    Shmulevich, I., & Kauffman, S. A. (2004). Activities and
    sensitivities in Boolean network models. Physical Review
    Letters, 93(4), 48701.
    http://doi.org/10.1103/PhysRevLett.93.048701

    The sensitivity of a Boolean function f on state vector x is the number of Hamming neighbors of x on which the function value is different than on x.

    The average sensitivity is an average taken over initial states.

    .. rubric:: Examples

    ::

        >>> from neet.boolean.examples import s_pombe
        >>> average_sensitivity(s_pombe)
        6.0
        >>> average_sensitivity(s_pombe,states=[[0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0]])
        5.5
        >>> average_sensitivity(s_pombe,states=[[0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0]],weights=[0.9,0.1])
        3.75
        >>> average_sensitivity(s_pombe,states=[[0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0,1,0]],weights=[9,1])
        3.75

    net    : NEET boolean network
    states : Optional list or generator of states.  If None, all states are used.
    weights: Optional list or generator of weights for each state.  
             If None, each state is equally weighted.
    """

    if not is_boolean_network(net):
        raise(TypeError("net must be a boolean network"))

    if states is None:
        states = net.state_space().states()

    if weights is not None:
        # currently changes generators to lists when weights are given.
        # is there a way to avoid this?
        states = list(states)
        weights = list(weights)
        if len(states) != len(weights):
            raise(ValueError("Length of weights and states must match"))

    sensList = []
    for state in states:
        sensList.append(sensitivity(net, state))

    if weights is not None:
        sensList = 1. / np.sum(weights) * \
            np.array(weights) * np.array(sensList)

    return np.mean(sensList)
