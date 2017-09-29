# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_boolean_network
import numpy as np

from .synchronous import transitions
#from .statespace import encode,decode


def sensitivity(net, state, transitions=None):
    """
    Calculate Boolean network sensitivity, as defined in, e.g.,

        Shmulevich, I., & Kauffman, S. A. (2004). Activities and
        sensitivities in Boolean network models. Physical Review 
        Letters, 93(4), 48701.
        http://doi.org/10.1103/PhysRevLett.93.048701

    The sensitivity of a Boolean function f on state vector x is the number of Hamming neighbors of x on which the function value is different than on x.
    
    This calculates the average sensitivity over all N boolean functions, 
    where N is the size of net.

    .. rubric:: Examples

    ::

        >>> from neet.boolean.examples import s_pombe
        >>> sensitivity(s_pombe,[0,0,0,0,0,1,1,0,0])
        1.0

    net    : NEET boolean network
    state  : A single network state, represented as a list of node states
    """

    if not is_boolean_network(net):
        raise(TypeError("net must be a boolean network"))

    # list Hamming neighbors
    neighbors = hamming_neighbors(state)

    nextState = net.update(state)

    # count sum of differences found in neighbors of the original
    s = 0.
    for neighbor in neighbors:
        if transitions is not None:
            newState = transitions[_fast_encode(neighbor)]
        else:
            newState = net.update(neighbor)
        s += boolean_distance(newState, nextState)

    return s / net.size

def _fast_encode(state):
    """
    Quickly find encoding of a binary state.
    
    Same result as net.state_space().encode(state).
    """
    # see https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
    out = 0
    for bit in state[::-1]:
        out = (out << 1) | bit
    return out

def boolean_distance(state1,state2):
    """
    Boolean distance between two states.
    """
    out = 0
    for i in range(len(state1)):
        out += (state1[i]+state2[i])%2
    return out

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

def average_sensitivity(net, states=None, weights=None, calc_trans=True):
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

        >>> from neet.boolean.examples import c_elegans
        >>> average_sensitivity(c_elegans)
        1.265625
        >>> average_sensitivity(c_elegans,states=[[0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1]])
        1.5
        >>> average_sensitivity(c_elegans,states=[[0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1]],weights=[0.9,0.1])
        1.7
        >>> average_sensitivity(s_pombe,states=[[0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1]],weights=[9,1])
        1.7

    net    : NEET boolean network
    states : Optional list or generator of states.  If None, all states are used.
    weights: Optional list or generator of weights for each state.  
             If None, each state is equally weighted.
    """

    if not is_boolean_network(net):
        raise(TypeError("net must be a boolean network"))
    
    # optionally pre-calculate transitions
    if calc_trans:
        trans = list(transitions(net))
    else:
        trans = None

    if states is None:
        states = net.state_space()

    if weights is not None:
        # currently changes generators to lists when weights are given.
        # is there a way to avoid this?
        states = list(states)
        weights = list(weights)
        if len(states) != len(weights):
            raise(ValueError("Length of weights and states must match"))

    sensList = []
    for state in states:
        sensList.append(sensitivity(net, state, trans))

    if weights is not None:
        sensList = 1. * len(sensList) / np.sum(weights) * \
            np.array(weights) * np.array(sensList)

    return np.mean(sensList)
