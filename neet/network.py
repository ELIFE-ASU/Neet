"""
.. currentmodule:: neet.network

.. testsetup:: network

    from neet.automata import ECA
    from neet.network import *
    from neet.statespace import StateSpace

API Documentation
-----------------
"""
from abc import ABCMeta, abstractmethod
import networkx as nx
import six


@six.add_metaclass(ABCMeta)
class Network(object):
    def __init__(self, size, names=None, metadata=None):
        if not isinstance(size, int):
            raise TypeError("Network size is not an int")
        elif size < 1:
            raise ValueError("Network size is negative")

        if metadata is None:
            metadata = dict()
        elif not isinstance(metadata, dict):
            raise TypeError('metadata is not a dict')

        self._size = size
        self._metadata = metadata
        self.names = names

    @property
    def size(self):
        return self._size

    @property
    def metadata(self):
        return self._metadata

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, names):
        if names is not None:
            try:
                names = list(names)
            except TypeError:
                raise TypeError('names must be convertable to a list')

            if len(names) != self.size:
                raise ValueError('number of names does not match network size')

        self._names = names

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def _unsafe_update(self, state, index, pin, values, *args, **kwargs):
        pass

    def update(self, state, index=None, pin=None, values=None, *args, **kwargs):
        space = self.state_space()

        if state not in space:
            raise ValueError("the provided state is not in the network's state space")

        if index is not None:
            if index < 0 or index >= self.size:
                raise IndexError("index out of range")
            elif pin is not None and pin != []:
                raise ValueError("cannot provide both the index and pin arguments")
            elif values is not None and values != {}:
                raise ValueError("cannot provide both the index and values arguments")
        elif pin is not None and values is not None:
            for k in values.keys():
                if k in pin:
                    raise ValueError("cannot set a value for a pinned state")
        if values is not None:
            bases = [space.base for _ in range(space.ndim)] if space.is_uniform else space.base
            for key in values.keys():
                val = values[key]
                if val < 0 or val >= bases[key]:
                    raise ValueError("invalid state in values argument")

        return self._unsafe_update(state, index, pin, values, *args, **kwargs)

    @abstractmethod
    def neighbors_in(self, index, *args, **kwargs):
        pass

    @abstractmethod
    def neighbors_out(self, index, *args, **kwargs):
        pass

    def neighbors(self, index, direction='both', *args, **kwargs):
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

    def to_networkx_graph(self, labels='indices', *args, **kwargs):
        if labels == 'indices':
            edges = [(i, j) for i in range(self.size) for j in self.neighbors_out(i)]
        elif labels == 'names' and self.names is not None:
            names = self.names
            edges = [(names[i], names[j]) for i in range(self.size) for j in self.neighbors_out(i)]
        elif labels == 'names' and self.names is None:
            raise ValueError("network nodes do not have names")
        else:
            raise ValueError("labels argument must be 'names' or 'indices', got {}".format(labels))

        kwargs.update(self.metadata)
        return nx.DiGraph(edges, **kwargs)

    def draw(self, filename=None, *args, **kwargs):
        graph = self.to_networkx_graph(*args, **kwargs)
        nx.nx_agraph.view_pygraphviz(graph, prog='circo', path=filename)
