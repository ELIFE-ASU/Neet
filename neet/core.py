# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import sys

def is_network(thing):
    """
    Determine whether an *object* or *type* meets the interface requirement of
    a network.

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
        >>> is_network(IsNetwork)
        True
        >>> is_network(IsNotNetwork())
        False
        >>> is_network(IsNotNetwork)
        False
        >>> is_network(5)
        False

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` has the minimum interface of a network
    """
    return hasattr(thing, 'update')

def is_fixed_sized(thing):
    """
    Determine whether an *object* or *type* is a network and has a fixed size.

    .. rubric:: Example

    ::

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...
        >>> class FixedSized(IsNetwork):
        ...     def size():
        ...         return 5
        ...
        >>> is_fixed_sized(IsNetwork)
        False
        >>> is_fixed_sized(FixedSized)
        True

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` is a network with a size attribute
    :see: :func:`is_network`.
    """
    return is_network(thing) and hasattr(thing, 'size')

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


def states(spec, b=2):
    """
    Generate all possible network states according to some specification,
    ``spec``.

    If ``spec`` is an integer, then it is taken to be the number of nodes in a
    network and ``b`` is assumed to be the base of the all of the nodes. As
    such, this function generates all base-``b`` sequences of length ``spec``.

    .. rubric:: Example:

    ::

        >>> list(neet.states(1))
        [[0], [1]]
        >>> list(neet.states(2))
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> list(neet.states(3))
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1
        , 0, 1], [0, 1, 1], [1, 1, 1]]

    ::

        >>> list(neet.states(2, b=1))
        [[0, 0]]
        >>> list(neet.states(2, b=2))
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> list(neet.states(2, b=3))
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1
        , 2], [2, 2]]

    If, however, ``spec`` is a list of integers, then each integer is assumed to
    be the base of some node in the network. The second second argument ``b``
    is ignored. In this case, it generates all sequences of length ``len(spec)``
    where the base of element ``i`` is ``spec[i]``.

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

    :param spec: the number of nodes or an array of node bases
    :type spec: int or list
    :param b: the base of the network nodes (ignored is ``spec`` if an list)
    :yields: a possible network state
    :raises TypeError: if ``spec`` is neither an int nor a list of ints
    """
    if isinstance(spec, int):
        if not isinstance(b, int):
            raise(TypeError("base must be an int"))

        for state in states([b]*spec):
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
