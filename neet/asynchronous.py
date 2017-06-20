# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import copy
from .interfaces import is_network, is_fixed_sized

def transitions(net, size=None, require_update=False):
    """
    Compute the asynchronous updates for a network.

    :param net: the network
    :type net: network
    :param size: the size of the network or None (if network is fixed-sized)
    :type size: int or None
    :param require_update: whether or not to require a node to update
    :type require_update: bool
    :yields: a dictionary with encoded states and probabilities as keys and values
    """
    if not is_network(net):
        raise TypeError("net must adhere to NEET's network interface")
    if is_fixed_sized(net):
        state_space = net.state_space()
        size = net.size
    elif size is not None:
        state_space = net.state_space(size)
    else:
        raise ValueError("must provide a size if network is variable-sized")

    for state in state_space.states():
        count = 0
        # A dictionary will not work when we admit non-encoded network states
        next_states = dict()
        for node in range(size):
            next_state = copy.copy(state)
            net.update(next_state, index=node)
            state_code = state_space.encode(next_state)
            if (not require_update) or next_state[node] != state[node]:
                if state_code in next_states:
                    next_states[state_code] += 1.0
                else:
                    next_states[state_code] = 1.0
                count += 1

        for state_code in next_states:
            next_states[state_code] /= count

        yield next_states
