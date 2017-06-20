# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import copy
from .interfaces import is_network, is_fixed_sized

def transitions(net):
    """
    Compute the asynchronous updates for a network.

    The implementation only provides support for fixed sized networks and
    includes all state updates, regardless of whether or not the update changed
    the state of the node in question.

    :param net: the network
    :type net: network
    :yields: a dictionary with encoded states and probabilities as keys and values
    """
    if not is_network(net):
        raise TypeError("net must adhere to NEET's network interface")
    # We need support for variable sized networks
    if not is_fixed_sized(net):
        raise ValueError("network must have a fixed size")

    state_space = net.state_space()
    size = net.size
    # This will need to change when we consider unchanging node states
    k = 1.0 / size
    for state in state_space.states():
        # A dictionary will not work when we admit non-encoded network states
        next_states = dict()
        for node in range(size):
            next_state = copy.copy(state)
            net.update(next_state, index=node)
            state_code = state_space.encode(next_state)
            if state_code in next_states:
                next_states[state_code] += k
            else:
                next_states[state_code] = k
        yield next_states
