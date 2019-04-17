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
and :func:`is_boolean_network` functions.

API Documentation
-----------------
"""
import six
from abc import ABCMeta, abstractmethod
from .statespace import StateSpace


@six.add_metaclass(ABCMeta)
class Network(object):
    def __init__(self, size):
        if not isinstance(size, int):
            raise TypeError("Network size is not an int")
        elif size < 1:
            raise ValueError("Network size is negative")
        self._size = size

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @property
    def size(self):
        return self._size

    @abstractmethod
    def neighbors_in(self, index, *args, **kwargs):
        pass

    @abstractmethod
    def neighbors_out(self, index, *args, **kwargs):
        pass

    def neighbors(self, index, direction='both', *args, **kwargs):
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

        if direction == 'in':
            return self.neighbors_in(index, *args, **kwargs)
        elif direction == 'out':
            return self.neighbors_out(index, *args, **kwargs)
        else:
            inputs = self.neighbors_in(index, *args, **kwargs)
            outputs = self.neighbors_out(index, *args, **kwargs)
            return inputs.union(outputs)


class BooleanNetwork(Network):
    def __init__(self, size):
        super(BooleanNetwork, self).__init__(size)
        self._state_space = StateSpace(self.size, base=2)

    def state_space(self):
        return self._state_space


Network.register(BooleanNetwork)


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
