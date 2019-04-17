"""
.. currentmodule:: neet.interfaces

.. testsetup:: interfaces

    from neet.automata import ECA
    from neet.interfaces import *
    from neet.statespace import StateSpace

Interfaces
==========

The :mod:`neet.interfaces` module provides a collection of functions for
determining if types adhere to various network interfaces, and generic
functions for operating upon them. This done primarily through the
:func:`is_fixed_sized` and :func:`is_boolean_network` functions.

API Documentation
-----------------
"""
import six
from abc import ABCMeta, abstractmethod, abstractproperty


@six.add_metaclass(ABCMeta)
class Network(object):
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def neighbors(self):
        pass

    @abstractproperty
    def size(self):
        pass


def is_fixed_sized(thing):
    """
    Determine whether an *object* is a network and has a fixed size.

    .. rubric:: Example

    .. doctest:: interfaces

        >>> class IsNetwork(Network):
        ...     def update(self):
        ...         pass
        ...     def state_space(self):
        ...         return StateSpace(1)
        ...     def neighbors(self, i):
        ...         return []
        ...
        >>> class FixedSized(IsNetwork):
        ...     def size():
        ...         return 5
        ...
        >>> is_fixed_sized(IsNetwork())
        False
        >>> is_fixed_sized(FixedSized())
        True

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` is a network with a size attribute
    """
    return isinstance(thing, Network) and hasattr(thing, 'size')


def is_boolean_network(thing):
    """
    Determine whether an *object* is a network with all Boolean states.d
    """
    # Boolean networks have a single base equal to 2
    if isinstance(thing, Network) and hasattr(thing.state_space(), 'base'):
        return thing.state_space().base == 2
    else:
        return False


def neighbors(net, index, direction='both', **kwargs):
    """
    Return a set of neighbors for a specified node, or a list of sets of
    neighbors for all nodes in the network.

    For ECAs, in the cases of the lattices having fixed boundary conditions,
    the left boundary, being on the left of the leftmost index 0, has an index
    of -1, while the right boundary's index is the size+1. The full state of
    the lattices and the boundaries is equavolent to: `[cell0, cell1, ...,
    cellN, right_boundary, left_boundary]` if it is ever presented as a single
    list in Python.

    :param index: node index, if neighbors desired for one node only
    :param direction: type of node neighbors to return ('in', 'out', or 'both')
    :kwarg size: size of ECA, required if network is an ECA
    :returns: a set of neighbors of a node

    .. rubric:: Example

    .. doctest:: interfaces

        >>> net = ECA(30)
        >>> neighbors(net, index=2, size=3, direction='out')
        {0, 1, 2}
        >>> net.boundary = (1,1)
        >>> neighbors(net, index=2, size=3, direction='out')
        {1, 2}

    See `ECA.neighbors()`,`LogicNetwork.neighbors()` or `WTNetwork.neighbors()`
    docstrings for more details and basic use examples.

    """
    if direction not in ('in', 'out', 'both'):
        raise ValueError('direction must be "in", "out" or "both"')

    neighbor_types = {'in': net.neighbors_in,
                      'out': net.neighbors_out,
                      'both': net.neighbors}

    return neighbor_types[direction](index)


def to_networkx_graph(net, size=None, labels='indices', **kwargs):
    """
    Return networkx graph given neet network. Requires `networkx`.

    :param labels: how node is labeled and thus identified in networkx graph
                   ('names' or 'indices'), only used if `net` is a
                   `LogicNetwork` or `WTNetwork`
    :returns: a networkx DiGraph
    """
    if net.__class__.__name__ == 'ECA':
        return net.to_networkx_graph()
    elif net.__class__.__name__ in ['WTNetwork', 'LogicNetwork']:
        return net.to_networkx_graph(labels=labels)
