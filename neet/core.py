# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import sys

def is_network(obj):
    """
    Determine whether an *object* meets the interface requirement of a network.

    .. rubric:: Example:

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class IsNotNetwork(object):
        ...     pass
        ...
        >>> is_network(IsNetwork())
        True
        >>> is_network(IsNotNetwork())
        False
        >>> is_network(5)
        False

    :param obj: an object
    :returns: ``True`` if ``obj`` is not a type and qualifies as a network
    """
    return not isinstance(obj, type) and hasattr(obj, 'update')


def is_network_type(cls):
    """
    Determine whether a *type* meets the interface requirement of a network.

    .. rubric:: Example:

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class IsNotNetwork(object):
        ...     pass
        ...
        >>> is_network_type(IsNetwork)
        True
        >>> is_network_type(IsNotNetwork)
        False
        >>> is_network_type(int)
        False

    :param cls: a class
    :returns: ``True`` if ``cls`` is a type and qualifies as a network
    """
    return isinstance(cls, type) and hasattr(cls, 'update')


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


def states(spec):
    """
    Generate all possible network states according to some specification,
    ``spec``.

    If ``spec`` is an integer, then it is taken to be the number of nodes in a
    boolean network. As such, it generates all boolean sequences of length
    ``spec``.

    .. rubric:: Example:

    ::

        >>> list(neet.states(1))
        [[0], [1]]
        >>> list(neet.states(2))
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> list(neet.states(3))
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1
        , 0, 1], [0, 1, 1], [1, 1, 1]]

    If, however, ``spec`` is a list of integers, then each integer is assumed to
    be the base of some node in the network. In this case, it generates all
    sequences of length ``len(spec)`` where the base of element ``i`` is
    ``spec[i]``.

    .. rubric:: Example:

    ::

        >>> list(neet.states([]))
        [[]]
        >>> list(neet.states([1]))
        [[0]]
        >>> list(neet.states([2]))
        [[0], [1]]
        >>> list(neet.states([3]))
        [[0], [1], [2]]
        >>> list(neet.states([2,3]))
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
        >>> list(neet.states([3,3]))
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2]
        , [1, 2], [2, 2]]

    :param spec: the number of boolean nodes or an array of node bases
    :type spec: int or list
    :yields: a possible network state
    :raises TypeError: if ``spec`` is neither an int nor a list of ints
    """
    if isinstance(spec, int):
        for state in states([2]*spec):
            yield state
    else:
        for i in range(len(spec)):
            if not isinstance(spec[i], int):
                raise(TypeError("spec is not an int nor a list of ints"))
        n = len(spec)
        state = [0]*n
        yield state[:]
        i = 0
        while i != n:
            if state[i] + 1 < spec[i]:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1
