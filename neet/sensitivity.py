# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_boolean_network
import numpy as np
import numpy.linalg as linalg
import copy

from .synchronous import transitions


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

def difference_matrix(net, state, transitions=None):
    """
    Returns matrix answering the question:
    Starting at the given state, does flipping the state of node j
    change the state of node i?
    """
    # set up empty matrix
    N = len(state)
    Q = np.empty((N,N))
    
    # list Hamming neighbors (in order!)
    neighbors = hamming_neighbors(state)

    nextState = net.update(state)

    # count differences found in neighbors of the original
    s = 0.
    for j,neighbor in enumerate(neighbors):
        if transitions is not None:
            newState = transitions[_fast_encode(neighbor)]
        else:
            newState = net.update(neighbor)
        Q[:,j] = [ ( nextState[i] + newState[i] )%2 for i in range(N) ]
        
    return Q

def _states_limited(nodes,state0):
    """
    All possible states that vary only nodes with given indices.
    """
    if len(nodes) == 0:
        return [state0]
    for i in nodes:
        stateFlipped = copy.copy(state0)
        stateFlipped[nodes[0]] = (stateFlipped[nodes[0]]+1)%2
        return _states_limited(nodes[1:],state0) + _states_limited(nodes[1:],stateFlipped)

def connections(net,nodei):
    return net.table[nodei][0]

def average_difference_matrix(net,states=None,weights=None,calc_trans=True):
    """
    Averaged over states, what is the probability
    that node i's state is changed by a single bit flip of node j?
    
    states (None)       : If None, average over all possible states. (For logic
                          networks, this case uses an algorithm that makes use of
                          sparse connectivity to be much more efficient.)  
                          Otherwise, providing a list of states will calculate 
                          the average over only those states.
    calc_trans (True)   : Optionally pre-calculate all transitions.  Only used
                          when states argument is not None.
    """
    N = net.size
    Q = np.zeros((N,N))

    if (states is not None) or (not hasattr(net,'table')):
        # optionally pre-calculate transitions
        if calc_trans:
            trans = list(transitions(net))
        else:
            trans = None
    
        # currently changes state generators to lists.
        # is there a way to avoid this?
        if states is None:
            states = list(net.state_space())

        if weights is not None:
            states = list(states)
            weights = list(weights)
            if len(states) != len(weights):
                raise(ValueError("Length of weights and states must match"))
        else:
            weights = np.ones_like(states)

        norm = np.sum(weights)
        for i,state in enumerate(states):
            Q += weights[i] * difference_matrix(net, state, trans) / norm

    else: # make use of sparse connectivity to be more efficient
        state0 = np.zeros(N,dtype=int)

        for i in range(N):
            nodesInfluencingI = connections(net,i)
            for jindex,j in enumerate(nodesInfluencingI):
            
                # for each state of other nodes, does j matter?
                otherNodes = list(copy.copy(nodesInfluencingI))
                otherNodes.pop(jindex)
                otherNodeStates = _states_limited(otherNodes,state0)
                for state in otherNodeStates:
                    # (might be able to do faster by calculating transitions once
                    #  for each i)
                    # (also we only need the update for node i)
                    # start with two states, one with j on and one with j off
                    jOff = copy.copy(state)
                    jOff[j] = 0
                    jOffNext = net.update(jOff)[i]
                    jOn = copy.copy(state)
                    jOn[j] = 1
                    jOnNext = net.update(jOn)[i]
                    # are the results different?
                    Q[i,j] += (jOffNext + jOnNext)%2
                Q[i,j] /= float(len(otherNodeStates))
            
    return Q

def lambdaQ(net,**kwargs):
    """
    Calculate sensitivity eigenvalue, the largest eigenvalue of the sensitivity
    matrix average_difference_matrix.
    """
    Q = average_difference_matrix(net,**kwargs)
    return max(np.sort(abs(linalg.eigvals(Q))))

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
             If None, each state is equally weighted.  If states and weights are
             both None, an algorithm is used to efficiently make use of sparse
             connectivity.
    """

    if not is_boolean_network(net):
        raise(TypeError("net must be a boolean network"))
    
    Q = average_difference_matrix(net,states=states,weights=weights,
                                  calc_trans=calc_trans)
    
    return np.sum(Q)/net.size
    
#        # optionally pre-calculate transitions
#        if calc_trans:
#            trans = list(transitions(net))
#        else:
#            trans = None
#
#        if states is None:
#            states = net.state_space()
#
#        if weights is not None:
#            # currently changes generators to lists when weights are given.
#            # is there a way to avoid this?
#            states = list(states)
#            weights = list(weights)
#            if len(states) != len(weights):
#                raise(ValueError("Length of weights and states must match"))
#
#        sensList = []
#        for state in states:
#            sensList.append(sensitivity(net, state, trans))
#
#        if weights is not None:
#            sensList = 1. * len(sensList) / np.sum(weights) * \
#                np.array(weights) * np.array(sensList)
#
#        return np.mean(sensList)
