# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_network, is_fixed_sized
from .states import StateSpace

import numpy as np

def trajectory(net, state, n=1):
    """
    Compute the trajectory of length ``n+1`` through state-space, as determined
    by the network rule, beginning at ``state``.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> rule30 = ECA(30)
        >>> trajectory(rule30, [0,0,1,0,0], n=5)
        array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 0, 0, 1],
               [0, 0, 1, 1, 1],
               [1, 1, 1, 0, 0],
               [1, 0, 0, 1, 1]])

    :param net: the network
    :param state: the network state
    :param n: the number of steps in the trajectory
    :returns: a ``numpy.ndarray`` of ``n+1`` network states
    :raises TypeError: ``not is_network(net)``
    :raises ValueError: if ``n < 1``
    """
    if not is_network(net):
        raise(TypeError("net is not a network"))
    if n < 1:
        raise(ValueError("number of steps must be positive, non-zero"))
    trajectory = [np.copy(state)]
    for i in range(n):
        trajectory.append(np.copy(trajectory[-1]))
        net.update(trajectory[-1])
    return np.asarray(trajectory)

def transitions(net, n=None, encode=True):
    """
    Generate the one-step state transitions for a network over its state space.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> gen = transitions(ECA(30), n=3)
        >>> gen
        <generator object transitions at 0x000001DF6E02B938>
        >>> list(gen)
        [0, 7, 7, 1, 7, 4, 2, 0]
        >>> list(transitions(ECA(30), n=3, encode=False))
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0,
        1], [0, 1, 0], [0, 0, 0]]

    :param net: the network
    :param n: the number of nodes in the network
    :type n: ``None`` or ``int``
    :param encode: encode the states as integers
    :type encode: boolean
    :yields: the one-state transitions
    :raises TypeError: if ``net`` is not a network
    :raises TypeError: if ``space`` is not a :class:`neet.StateSpace`
    """
    if not is_network(net):
        raise(TypeError("net is not a network"))

    if is_fixed_sized(net):
        if n is not None:
            raise(TypeError("n must be None for fixed sized networks"))
        space = net.state_space()
    else:
        space = net.state_space(n)

    if not isinstance(space, StateSpace):
        raise(TypeError("network's state space is not an instance of StateSpace"))

    for state in space.states():
        net.update(state)
        if encode:
            yield space.encode(state)
        else:
            yield state
