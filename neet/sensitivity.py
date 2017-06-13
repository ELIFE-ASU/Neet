# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from interfaces import is_boolean_network
import numpy as np

def sensitivity(net, state):

    """
    Calculate Boolean network sensitivity, as defined in, e.g.,
    
        Shmulevich, I., & Kauffman, S. A. (2004). Activities and
        sensitivities in Boolean network models. Physical Review 
        Letters, 93(4), 48701.
        http://doi.org/10.1103/PhysRevLett.93.048701
        
    The sensitivity of a Boolean function f on state vector x is the number of Hamming neighbors of x on which the function value is different than on x. This should be straightforward to implement for synchronous networks.
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
        if not np.array_equal(newState,nextState):
            s += 1

    return s
        
def hamming_neighbors(state):
    """
    Return Hamming neighbors of a boolean state.
    
    .. rubric:: Examples
    
    ::
    
        >>> hamming_neighbors([0,0,1])
        
    """
    state = np.asarray(state, dtype=int)
    if len(state.shape) > 1:
        raise(ValueError("state must be 1-dimensional"))
    if not np.array_equal(state%2, state):
        raise(ValueError("state must be binary"))

    repeat = np.tile(state,(len(state),1))
    neighbors = (repeat + np.diag(np.ones_like(state)))%2
    
    return neighbors


