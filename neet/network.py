"""
.. currentmodule:: neet.network

.. testsetup:: network

    from neet.boolean import ECA
    from neet.network import *
    from neet.statespace import StateSpace

API Documentation
-----------------
"""
from abc import ABCMeta, abstractmethod
from .python import long
from .statespace import StateSpace
from .landscape import LandscapeMixin
import networkx as nx
import six


@six.add_metaclass(ABCMeta)
class Network(LandscapeMixin, StateSpace):
    """
    The ``Network`` class represents the core of Neet's functionality of 
    simulating and analyzing network models, such as gene-regulatory networks.
    The ``Network`` class currently supports simulating synchronous Boolean 
    network models, though the API is designed to be model generic. Future 
    work will implement asynchronous update mechanisms and more general network 
    types.
    """
    def __init__(self, shape, names=None, metadata=None):
        super(Network, self).__init__(shape)

        if metadata is None:
            metadata = dict()
        elif not isinstance(metadata, dict):
            raise TypeError('metadata is not a dict')

        self._metadata = metadata
        self.names = names

    @property
    def metadata(self):
        return self._metadata

    @property
    def names(self):
        """
        The names of nodes in the network.

        .. rubric:: Examples

        .. doctest:: network

            >>> s_pombe.names
            ['SK',
            'Cdc2_Cdc13',
            'Ste9',
            'Rum1',
            'Slp1',
            'Cdc2_Cdc13_active',
            'Wee1_Mik1',
            'Cdc25',
            'PP']
            >>> s_pombe.names = ["name_"+str(i) for i in range(len(s_pombe.names))]
            ['name_0',
            'name_1',
            'name_2',
            'name_3',
            'name_4',
            'name_5',
            'name_6',
            'name_7',
            'name_8']
        
        :param names: list of node names to use for network (length must equal the number of nodes).
        :return: list of node names
        """
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
    def _unsafe_update(self, state, index, pin, values, *args, **kwargs):
        pass

    def update(self, state, index=None, pin=None, values=None, *args, **kwargs):
        """
        Updates the network from its current state to its next state.

        Refer to documentation for `_unsafe_update` for usage.
        """
        if state not in self:
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
            bases = self.shape
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
        """
        Return the set of all neighbor nodes, where either 
        edge(neighbor_node-->index) and/or edge(index-->neighbor_node) exists.

        Calls the `neighbors_in` and/or `neighbors_out` methods of the specific
        network type.

        .. rubric:: Examples

        .. doctest:: network

            >>> net = WTNetwork([[0,0,0],[1,0,1],[0,1,0]],
            ... theta=WTNetwork.split_threshold)
            >>> net.neighbors(0)
            {0, 1}
            >>> [net.neighbors_in(node) for node in range(net.size)]
            [{0, 1}, {0, 1, 2}, {1, 2}]
        
        :param index: node index
        :param direction: `in`coming or `out`going neighbors, or `both`
        :param args: arguments passed to network's `neighbors_in` and/or `neighbors_out`
        :param kwargs: arguments passed to network's `neighbors_in` and/or `neighbors_out`
        :returns: the set of all node indices which point toward the index node
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

    def network_graph(self, labels='indices', **kwargs):
        """
        The graph of the network as a ``networkx.Digraph``.

        .. rubric:: Examples

        .. doctest:: network

            >>> s_pombe.network_graph()
            <networkx.classes.digraph.DiGraph object at 0x106504810>
        
        :param labels: label to be applied to graph nodes (either `indices` or `names`)
        :param kwargs: kwargs to pass to `nx.DiGraph`
        :return: a networkx DiGraph object
        """
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

    def draw_network_graph(self, graphkwargs={}, pygraphkwargs={}):
        """
        Draw network's networkx graph using PyGraphviz.

        Requires graphviz (cannot be installed via pip--see:
        https://graphviz.gitlab.io/download/) and pygraphviz
        (can be installed via pip).

        .. rubric:: Examples

        .. doctest:: network

            >>> s_pombe.draw_network_graph()

        :param graphkwargs: kwargs to pass to `network_graph`
        :param pygraphkwargs: kwargs to pass to `view_pygraphviz`
        """
        from .draw import view_pygraphviz
        default_args = { 'prog': 'circo' }
        graph = self.network_graph(**graphkwargs)
        view_pygraphviz(graph, **dict(default_args, **pygraphkwargs))

class UniformNetwork(Network):
    def __init__(self, size, base, names=None, metadata=None):
        super(UniformNetwork, self).__init__([base] * size, names, metadata)
        self._base = base

    @property
    def base(self):
        return self._base

    def __iter__(self):
        size, base = self.size, self.base
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            if state[i] + 1 < base:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, state):
        try:
            if len(state) != self.size:
                return False

            base = self.base
            for x in state:
                if x < 0 or x >= base:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        base = self.base
        for x in state:
            encoded += place * long(x)
            place *= base

        return encoded

    def decode(self, encoded):
        size, base = self.size, self.base
        state = [0] * size
        for i in range(size):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state


Network.register(UniformNetwork)
