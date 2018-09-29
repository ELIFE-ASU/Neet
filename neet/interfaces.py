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
:func:`is_network`, :func:`is_fixed_sized` and :func:`is_boolean_network`
functions.

API Documentation
-----------------
"""


def is_network(thing):
    """
    Determine whether an *object* or *type* meets the interface requirement of
    a network. Specifically, to be considered a network, a class must provide
    the following methods:

    1. `update` which updates the state of a lattice
    2. `state_space` which returns a :func:`neet.statespace.StateSpace` object
    3. `neighbors` which returns the neighbors of a given node

    .. rubric:: Example:

    .. doctest:: interfaces

        >>> class IsNetwork(object):
        ...     def update(self):
        ...         pass
        ...     def state_space(self):
        ...         return StateSpace(1)
        ...     def neighbors(self, i):
        ...         return []
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
    has_update = hasattr(thing, 'update')
    has_state_space = hasattr(thing, 'state_space')
    has_neighbors = hasattr(thing, 'neighbors')

    return has_update and has_state_space and has_neighbors


def is_fixed_sized(thing):
    """
    Determine whether an *object* or *type* is a network and has a fixed size.

    .. rubric:: Example

    .. doctest:: interfaces

        >>> class IsNetwork(object):
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
        >>> is_fixed_sized(IsNetwork)
        False
        >>> is_fixed_sized(FixedSized)
        True

    :param thing: an object or a type
    :returns: ``True`` if ``thing`` is a network with a size attribute
    :see: :func:`is_network`.
    """
    return is_network(thing) and hasattr(thing, 'size')


def is_boolean_network(thing):
    """
    Determine whether an *object* is a network with all Boolean states.d
    """
    # Boolean networks have a single base equal to 2
    if is_network(thing) and hasattr(thing.state_space(), 'base'):
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

    if net.__class__.__name__ == 'ECA':
        if 'size' not in kwargs:
            msg = "A `size` kwarg is required for returning an ECA's neighbors"
            raise AttributeError(msg)
        else:
            return neighbor_types[direction](index, size=kwargs['size'])

    else:
        return neighbor_types[direction](index)


def to_networkx_graph(net, size=None, labels='indices', **kwargs):
    """
    Return networkx graph given neet network. Requires `networkx`.

    :param labels: how node is labeled and thus identified in networkx graph
                   ('names' or 'indices'), only used if `net` is a
                   `LogicNetwork` or `WTNetwork`
    :kwarg size: size of ECA, required if network is an ECA
    :returns: a networkx DiGraph
    """
    if net.__class__.__name__ == 'ECA':
        if size is None:
            msg = "`size` required to convert an ECA to a networkx network"
            raise AttributeError(msg)
        else:
            return net.to_networkx_graph(size)

    elif net.__class__.__name__ in ['WTNetwork', 'LogicNetwork']:
        return net.to_networkx_graph(labels=labels)
