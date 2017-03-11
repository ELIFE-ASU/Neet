# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

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

