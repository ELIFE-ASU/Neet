"""
.. currentmodule:: neet

.. testsetup:: network

    from neet.boolean import ECA
    from neet.boolean.examples import s_pombe

The :mod:`neet` module provides the following abstract network classes from
which all concrete Neet networks inherit:

.. autosummary::
    :nosignatures:

    Network
    UniformNetwork

.. inheritance-diagram:: neet.Network neet.UniformNetwork
   :parts: 1

These classes provide an abstract interface which algorithms can leverage for
generic implementation of various network-theoretic analyses.
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
    The Network class is the core base class for all Neet networks. It provides
    an interface for describing network state updating and simple
    graph-theoretic analyses.

    .. autosummary::
        :nosignatures:

        names
        metadata
        _unsafe_update
        update
        neighbors_in
        neighbors_out
        neighbors
        network_graph
        draw_network_graph

    Network is an *abstract* class, meaning it cannot be instantiated, and
    inherits from :class:`neet.LandscapeMixin` and :class:`neet.StateSpace`.
    Initialization of the Network requires, at a minimum, a specification of
    the shape of the network's state space, and optionally allows the user to
    specify a list of names for the nodes of the network and a metadata
    dictionary for the network as a whole (e.g. citation information).

    Any concrete deriving class must overload the following methods:

    * :meth:`_unsafe_update`
    * :meth:`neighbors_in`
    * :meth:`neighbors_out`

    :param shape: the base of each node of the network
    :type shape: list
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
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
        """
        Any metadata associated with the network.
        """
        return self._metadata

    @property
    def names(self):
        """
        Get or set the names of the nodes of the network.

        :raises TypeError: if the assigned value is not convertable to a list
        :raises ValueError: if the length fo the assigned values does not match the networks's size
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
        """
        Unsafely update the state of a network in place.

        This function accepts three optional arguments by default:

        * ``index``  - update only the specified node (by index)
        * ``pin``    - do not update the state of any node in a list
        * ``values`` - set the state of some subset of nodes to specified values

        .. Note::

            As an abstract method, every concrete class derving from Network
            must overload this method. The overload **should not** perform no
            ensurance checks on the arguments to maximize performance, as those
            check are performed in the :meth:`update` method. Further, it is
            assumed that this method *modifies* the ``state`` argument in-place
            and no others.

        :param state: the state of the network to update
        :type state: list, numpy.ndarray
        :param index: the index to update
        :type index: int or None
        :param pin: which nodes to pin to their current state
        :type pin: list, numpy.ndarray or None
        :param values: a dictionary mapping nodes to a state to which to reset the node to
        :type values: dict or None
        :returns: the updated state
        """
        pass

    def update(self, state, index=None, pin=None, values=None, *args, **kwargs):
        """
        Update the state of a network in place.

        This function accepts three optional arguments by default:

        * ``index``  - update only the specified node (by index)
        * ``pin``    - do not update the state of any node in a list
        * ``values`` - set the state of some subset of nodes to specified values

        .. rubric:: Examples

        **Updates States In-Place:**

        .. doctest:: network

            >>> rule = ECA(30, size=5)
            >>> state = [0, 0, 1, 0, 0]
            >>> rule.update(state)
            [0, 1, 1, 1, 0]
            >>> state
            [0, 1, 1, 1, 0]

        **Updating A Single Node:**

        .. doctest:: network

            >>> rule = ECA(30, size=5)
            >>> rule.update([0, 0, 1, 0, 0])
            [0, 1, 1, 1, 0]
            >>> rule.update([0, 0, 1, 0, 0], index=1)
            [0, 1, 1, 0, 0]

        **Pinning States:**

        .. doctest:: network

            >>> rule = ECA(30, size=5)
            >>> rule.update([0, 0, 1, 0, 0])
            [0, 1, 1, 1, 0]
            >>> rule.update([0, 0, 1, 0, 0], pin=[1])
            [0, 0, 1, 1, 0]


        **Overriding States:**

        .. doctest:: network

            >>> rule = ECA(30, size=5)
            >>> rule.update([0, 0, 1, 0, 0])
            [0, 1, 1, 1, 0]
            >>> rule.update([0, 0, 1, 0, 0], values={0: 1, 2: 0})
            [1, 1, 0, 1, 0]

        This function ensures that:

        1. If ``index`` is provided, then neither ``pin`` nor ``values`` is
           provided.
        2. If ``pin`` and ``values`` are both provided, then they do not affect
           the same nodes.
        3. If ``values`` is provided, then the overriding states specified in
           it are consistent with the state space of the network.

        .. Note::

            Typically, this method should not be overloaded unless the
            particular deriving class makes use of the ``args`` or ``kwargs``
            arguments. In that case, it should first ensure that those
            arguments are well-behaved, and and the delegate subsequent checks
            and the call to :meth:`_unsafe_update` to a call to this
            :meth:`neet.Network.update`.

        :param state: the state of the network to update
        :type state: list or numpy.ndarray
        :param index: the index to update
        :type index: int or None
        :param pin: which nodes to pin to their current state
        :type pin: list, numpy.ndarray or None
        :param values: a dictionary mapping nodes to a state to which to reset the node to
        :type values: dict or None
        :returns: the updated state
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
        """
        Get a set of all incoming neighbors of the node at ``index``.

        All concrete network classes must overload this method.

        :param index: the index of the node target node
        :type index: int
        :returns: a set of incoming neighbor indices
        """
        pass

    @abstractmethod
    def neighbors_out(self, index, *args, **kwargs):
        """
        Get a set of all outgoing neighbors of the node at ``index``.

        All concrete network classes must overload this method.

        :param index: the index of the node source node
        :type index: int
        :returns: a set of outgoing neighbor indices
        """
        pass

    def neighbors(self, index, direction='both', *args, **kwargs):
        """
        Get a set of the neighbors of the node at ``index``. Optionally,
        specify the directionality of the neighboring edges, e.g. ``'in'``,
        ``'out'`` or ``'both'``.

        .. rubric:: Examples

        **All Neighbors:**

        .. doctest:: network

            >>> s_pombe.neighbors(7)
            {1, 5, 7, 8}

        **Incoming Neighbors:**

        .. doctest:: network

            >>> s_pombe.neighbors(7, direction='in')
            {8, 1, 7}

        **Outgoing Neighbors:**

        .. doctest:: network

            >>> s_pombe.neighbors(7, direction='out')
            {5, 7}

        :param index: the index of the node
        :type index: int
        :param direction: the directionality of the neighboring edges
        :type direction: str
        :returns: a set of neighboring node indices, respecting ``direction``.
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
        The graph of the network as a :class:`networkx.DiGraph`.

        This method should only be overloaded by derived classes if additional
        metadata is to be added to the graph by default.

        .. rubric:: Examples

        .. doctest:: network

            >>> s_pombe.network_graph()
            <networkx.classes.digraph.DiGraph object at 0x...>

        :param labels: label to be applied to graph nodes (either ``'indices'`` or ``'names'``)
        :param kwargs: kwargs to pass to the :class:`networkx.DiGraph` constructor
        :return: a :class:`networkx.DiGraph` object
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

        .. Note::

            This method requires `Graphviz <https://graphviz.org/>`_ and
            `pygraphviz <https://pypi.org/project/pygraphviz/>`_. The former
            requires manual installation (see
            https://graphviz.gitlab.io/download/), while the latter can be
            installed via ``pip``.

        :param graphkwargs: kwargs to pass to :meth:`network_graph`
        :param pygraphkwargs: kwargs to pass to :func:`neet.draw.view_pygraphviz`
        """
        from .draw import view_pygraphviz
        default_args = {'prog': 'circo'}
        graph = self.network_graph(**graphkwargs)
        view_pygraphviz(graph, **dict(default_args, **pygraphkwargs))


class UniformNetwork(Network):
    """
    The UnformNetwork class represents a network in which every node has the
    same number of discrete states. This allows for more efficient default
    implementations of several methods. If your particular concrete network
    type meets this condition, then you should derive from UniformNetwork
    rather than Network.

    .. inheritance-diagram:: neet.UniformNetwork
       :parts: 1


    In addition to the methods provided by :class:`Network`, UniformNetwork
    also provides the following attribute:

    .. autosummary::
        :nosignatures:

        base

    UniformNetwork derives from :class:`Network`, but is still *abstract*,
    meaning it cannot be instantiated. Initialization of the
    :class:`UniformNetwork` requires, at a minimum, the number of nodes in the
    network (``size``) and the number of states the nodes can take (``base``).
    As with :class:`Network`, the user can optionally specify a list of names
    for the nodes of the network and a metadata dictionary for the network as a
    whole (e.g. citation information).

    Any concrete deriving class must overload the following methods:

    * :meth:`_unsafe_update`
    * :meth:`neighbors_in`
    * :meth:`neighbors_out`

    :param size: the number of nodes in the network
    :type size: int
    :param base: the number of states each node can take
    :type base: int
    :param names: an interable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    """

    def __init__(self, size, base, names=None, metadata=None):
        super(UniformNetwork, self).__init__([base] * size, names, metadata)
        self._base = base

    @property
    def base(self):
        """
        Get the number of states each node can take.

        .. rubric:: Examples

        .. doctest:: network

           >>> ECA(30, size=5).base
           2

        :returns: the base of nodes of the network
        """
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
