# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np

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
