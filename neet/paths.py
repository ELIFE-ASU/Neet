# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_network
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

def transitions(net, space, encode=True):
    """
    Generate the one-step state transitions for a network over a state space.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> gen = transitions(ECA(30), StateSpace(3))
        >>> gen
        <map object at 0x00000219ABAE86D8>
        >>> list(gen)
        [0, 7, 7, 1, 7, 4, 2, 0]
        >>> gen = transitions(ECA(30), StateSpace(3), encode=False)
        >>> list(gen)
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0,
        1], [0, 1, 0], [0, 0, 0]]

    :param net: the network
    :param space: the ``StateSpace`` for then network
    :param encode: encode the states as integers
    :type encode: boolean
    :returns: a generator over the one-state transitions
    :raises TypeError: if ``net`` is not a network
    :raises TypeError: if ``space`` is not a :class:`neet.StateSpace`
    """
    if not is_network(net):
        raise(TypeError("net is not a network"))
    if not isinstance(space, StateSpace):
        raise(TypeError("space is not a StateSpace"))
    states = map(net.update, space.states())
    if encode:
        return map(space.encode, states)
    else:
        return states
