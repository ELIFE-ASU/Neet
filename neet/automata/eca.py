"""
Elementary Cellular Automata
============================

The :class:`neet.automata.eca.ECA` class describes an `Elementary Cellular
Automaton <https://en.wikipedia.org/wiki/Elementary_cellular_automaton>`_
with an arbitrary rule. The ``ECA`` class is **not** a fixed sized network.
This means that the size is determined when it is used based on arguments
passed to the relevant methods or functions.

.. rubric:: Examples

.. doctest:: automata

    >>> ca = ECA(30)
    >>> ca.update([0, 0, 1, 0, 0])
    [0, 1, 1, 1, 0]
    >>> ca.update([0, 1, 0])
    [1, 1, 1]
    >>> transitions(ca, size=3)
    [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]]
"""
import numpy as np
import networkx as nx
from neet.statespace import StateSpace


class ECA(object):
    """
    ECA is a class to represent elementary cellular automaton rules. Each ECA
    contains an 8-bit integral member variable ``code`` representing the
    Wolfram code for the ECA rule and a set of boundary conditions which is
    either ``None``, signifying periodic boundary conditions, or a pair of
    cell states signifying fixed, open boundary conditions.
    """

    def __init__(self, code, boundary=None):
        """
        Construct an elementary cellular automaton rule.

        .. rubric:: Examples

        .. doctest:: automata

            >>> ca = ECA(30)
            >>> ca.code
            30
            >>> ca.boundary
            >>> ca = ECA(30, boundary=(0,0))
            >>> ca.boundary
            (0, 0)

        :param code: the Wolfram code for the ECA
        :type code: int
        :param boundary: the boundary conditions for the CA
        :type boundary: tuple or None
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\{0,1,\ldots,255\}`
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance
                           of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of
                            binary states
        """
        self.code = code
        self.boundary = boundary

    @property
    def code(self):
        """
        The Wolfram code of the elementary cellular automaton

        .. rubric:: Examples

        .. doctest:: automata

            >>> eca = ECA(30)
            >>> eca.code
            30
            >>> eca.code = 45
            >>> eca.code
            45
            >>> eca.code = 256
            Traceback (most recent call last):
                ...
            ValueError: invalid ECA code

        :type: int
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\{0,1,\ldots,255\}`
        """
        return self.__code

    @code.setter
    def code(self, code):
        if not isinstance(code, int):
            raise TypeError("ECA code is not an int")
        if 255 < code or code < 0:
            raise ValueError("invalid ECA code")
        self.__code = code

    @property
    def boundary(self):
        """
        The boundary conditions of the elemenary cellular automaton

        .. rubric:: Examples

        .. doctest:: automata

            >>> eca = ECA(30)
            >>> eca.boundary
            >>> eca.boundary = (0,1)
            >>> eca.boundary
            (0, 1)
            >>> eca.boundary = None
            >>> eca.boundary
            >>> eca.boundary = [0,1]
            Traceback (most recent call last):
                ...
            TypeError: ECA boundary are neither None nor a tuple

        :type: ``None`` or tuple
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance
                           of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of
                            binary states
        """
        return self.__boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary and not isinstance(boundary, tuple):
            raise TypeError("ECA boundary are neither None nor a tuple")
        if boundary:
            if len(boundary) != 2:
                raise ValueError("invalid ECA boundary conditions")
            for x in boundary:
                if x != 0 and x != 1:
                    raise ValueError("invalid ECA boundary value")
        self.__boundary = boundary

    def state_space(self, n):
        """
        Return a :class:`neet.statespace.StateSpace` object for a
        lattice of length ``n``.

        .. doctest:: automata

            >>> eca = ECA(30)
            >>> eca.state_space(3)
            <neet.statespace.StateSpace object at 0x...>
            >>> space = eca.state_space(3)
            >>> list(space)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :param n: the number of nodes in the lattice
        :type n: int
        :returns: :class:`neet.statespace.StateSpace`
        :raises ValueError: if ``n < 1``
        """
        return StateSpace(n, base=2)

    def _unsafe_update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice``, in place, without
        checking the validity of the arguments.

        .. rubric:: Basic Use:

        .. doctest:: automata

            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs)
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update([0,0,1,0,0])
            [1, 1, 1, 1, 1]

        .. rubric:: Single-Node Update:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, index=1)
            [0, 1, 1, 0, 0]
            >>> xs
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update(xs, index=-1)
            [0, 1, 1, 0, 1]

        .. rubric:: State Pinning:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, pin=[-2])
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update(xs, pin=[4])
            [0, 1, 0, 1, 0]

        .. rubric:: Value Fixing:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, values={0:1,-2:0})
            [1, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> xs = [1,1,1,0,0]
            >>> ca._unsafe_update(xs, values={1:0,-1:0})
            [0, 0, 0, 1, 0]

        :param lattice: the one-dimensional sequence of states
        :type lattice: sequence
        :param index: the index to update (or None)
        :param pin: a sequence of indicies to pin (or None)
        :param values: a dictionary of index-value pairs to fix after update
        :returns: the updated lattice
        """
        pin_states = pin is not None and pin != []
        if self.boundary:
            left = self.__boundary[0]
            right = self.__boundary[1]
        else:
            left = lattice[-1]
            right = lattice[0]
        code = self.code
        if index is None:
            if pin_states:
                pinned = np.asarray(lattice)[pin]
            temp = 2 * left + lattice[0]
            for i in range(1, len(lattice)):
                temp = 7 & (2 * temp + lattice[i])
                lattice[i - 1] = 1 & (code >> temp)
            temp = 7 & (2 * temp + right)
            lattice[-1] = 1 & (code >> temp)
            if pin_states:
                for (j, i) in enumerate(pin):
                    lattice[i] = pinned[j]
        else:
            if index < 0:
                index += len(lattice)

            if index == 0:
                temp = left
            else:
                temp = lattice[index - 1]

            temp = 2 * temp + lattice[index]

            if index + 1 == len(lattice):
                temp = 2 * temp + right
            else:
                temp = 2 * temp + lattice[index + 1]

            lattice[index] = 1 & (code >> (7 & temp))
        if values is not None:
            for key in values:
                lattice[key] = values[key]
        return lattice

    def update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice`` in place.

        .. rubric:: Basic Use:

        .. doctest:: automata

            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs)
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update([0,0,1,0,0])
            [1, 1, 1, 1, 1]

        .. rubric:: Single-Node Update:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, index=1)
            [0, 1, 1, 0, 0]
            >>> xs
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update(xs, index=-1)
            [0, 1, 1, 0, 1]

        .. rubric:: State Pinning:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, pin=[-2])
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update(xs, pin=[4])
            [0, 1, 0, 1, 0]

        .. rubric:: Value Fixing:

        .. doctest:: automata

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, values={0:1,-2:0})
            [1, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> xs = [1,1,1,0,0]
            >>> ca.update(xs, values={1:0,-1:0})
            [0, 0, 0, 1, 0]

        .. rubric:: Erroneous Usage:

        .. doctest:: automata

            >>> xs = []
            >>> ca.update(xs)
            Traceback (most recent call last):
            ...
            ValueError: lattice is empty
            >>> xs = [0,0,2,0,0]
            >>> ca.update(xs)
            Traceback (most recent call last):
            ...
            ValueError: invalid value "2" in lattice
            >>> ca.update(xs, index=5)
            Traceback (most recent call last):
            ...
            ValueError: the provided state is not in the ECA's state space
            >>> ca.update([0,0,1,0,0,], index=1, pin=[0])
            Traceback (most recent call last):
            ...
            ValueError: cannot provide both the index and pin arguments
            >>> ca.update([0,0,1,0,0], index=1, values={0:0})
            Traceback (most recent call last):
            ...
            ValueError: cannot provide both the index and values arguments
            >>> ca.update([0,0,1,0,0], pin=[2], values={2:0})
            Traceback (most recent call last):
            ...
            ValueError: cannot set a value for a pinned state
            >>> ca.update([0,0,1,0,0], values={2:2})
            Traceback (most recent call last):
            ...
            ValueError: invalid state in values argument

        :param lattice: the one-dimensional sequence of states
        :param index: the index to update (or None)
        :param pin: a sequence of indicies to pin (or None)
        :param values: a dictionary of index-value pairs to fix after update
        :returns: the updated lattice
        :raises ValueError: if ``lattice`` is not in the ECA's state space
        :raises IndexError: if ``index is not None and index > len(states)``
        :raises ValueError: if ``index`` and ``pin`` are both provided
        :raises ValueError: if ``index`` and ``values`` are both provided
        :raises ValueError: if an element of ``pin`` is a key in ``values``
        :raises ValueError: if a value in ``values`` is not binary (0 or 1)
        """
        size = len(lattice)
        if lattice not in self.state_space(size):
            msg = "the provided state is not in the ECA's state space"
            raise ValueError(msg)

        if index is not None:
            if index < -size:
                raise IndexError("lattice index out of range")
            elif pin is not None and pin != []:
                msg = "cannot provide both the index and pin arguments"
                raise ValueError(msg)
            elif values is not None and values != {}:
                msg = "cannot provide both the index and values arguments"
                raise ValueError(msg)
        elif pin is not None and values is not None:
            for key in values.keys():
                if key in pin:
                    raise ValueError("cannot set a value for a pinned state")
        if values is not None:
            for val in values.values():
                if val != 0 and val != 1:
                    raise ValueError("invalid state in values argument")

        return self._unsafe_update(lattice, index, pin, values)

    def neighbors_in(self, index, size, **kwargs):
        """
        Return the set of all incoming neighbor nodes.

        In the cases of the lattices having fixed boundary conditions, the
        left boundary, being on the left of the leftmost index 0, has an index
        of -1, while the right boundary's index is the size+1. The full state
        of the lattices and the boundaries is equavolent to: `[cell0, cell1,
        ..., cellN, right_boundary, left_boundary]` if it is ever presented as
        a single list in Python.

        :param index: node index
        :param size: size of ECA
        :returns: the set of all node indices which point toward the index node
        :raises ValueError: if `index < 0 or index > n - 1`

        .. rubric:: Basic Use:

        .. doctest:: automata

            >>> net = ECA(30)
            >>> net.neighbors_in(1, size=3)
            {0, 1, 2}
            >>> net.neighbors_in(2, size=3)
            {0, 1, 2}
            >>> net.boundary = (1,1)
            >>> net.neighbors_in(2, size=3)
            {1, 2, 3}
            >>> net.neighbors_in(0, 3)
            {0, 1, -1}

        .. rubric:: Erroneous Usage:

        .. doctest:: automata

            >>> net = ECA(30,boundary=(1, 1))
            >>> net.neighbors_in(5, 3)
            Traceback (most recent call last):
                ...
            ValueError: index must be a non-negative integer less than size
        """
        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")

        if size < 1:
            raise ValueError("size must be a positive integer")

        if not isinstance(index, int):
            raise TypeError("index must be a non-negative integer")

        if index < 0 or index > size - 1:
            msg = "index must be a non-negative integer less than size"
            raise ValueError(msg)

        left, right = index - 1, index + 1

        if left < 0 and self.boundary is None:
            left = size - 1

        if right > size - 1 and self.boundary is None:
            right = 0

        return {left, index, right}

    def neighbors_out(self, index, size):
        """
        Return the set of all outgoing neighbor nodes.

        Fixed boundaries are excluded as they are not affected by internal
        states.

        :param index: node index
        :param size: size of ECA
        :returns: the set of all node indices which point from the index node
        :raises ValueError: if `index < 0 or index > n - 1`

        .. rubric:: Basic Use:

        .. doctest:: automata

            >>> net = ECA(30)
            >>> net.neighbors_out(1, 3)
            {0, 1, 2}
            >>> net.neighbors_out(2, 3)
            {0, 1, 2}
            >>> net.boundary = (1, 1)
            >>> net.neighbors_out(2, 3)
            {1, 2}
            >>> net.neighbors_out(0, 3)
            {0, 1}

        .. rubric:: Erroneous Usage:

        .. doctest:: automata

            >>> net = ECA(30,boundary=(1, 1))
            >>> net.neighbors_out(5, 3)
            Traceback (most recent call last):
                ...
            ValueError: index must be a non-negative integer less than size
        """
        if not isinstance(index, int):
            raise TypeError("index must be a non-negative integer")

        if not isinstance(size, int):
            raise TypeError("size must be a positive integer")

        if size < 1:
            raise ValueError("size must be a positive integer")

        if index < 0 or index > size - 1:
            msg = "index must be a non-negative integer less than size"
            raise ValueError(msg)

        left, right = index - 1, index + 1

        if left < 0:
            left = size - 1 if self.boundary is None else 0

        if right > size - 1:
            right = 0 if self.boundary is None else size - 1

        return {left, index, right}

    def neighbors(self, index, size):
        """
        Return a set of neighbors for a specified node.

        In the cases of the lattices having fixed boundary conditions, the
        left boundary, being on the left of the leftmost index 0, has an index
        of -1, while the right boundary's index is the size+1. The full state
        of the lattices and the boundaries is equavolent to: `[cell0, cell1,
        ..., cellN, right_boundary, left_boundary]` if it is ever presented as
        a single list in Python.

        :param index: node index
        :param size: size of ECA
        :returns: the set of all node indices adjacent to the index node
        :raises ValueError: if `index < 0 or index > n - 1`

        .. rubric:: Basic Use:

        .. doctest:: automata

            >>> net = ECA(30)
            >>> net.neighbors(1, size=3)
            {0, 1, 2}
            >>> net.neighbors(2, size=3)
            {0, 1, 2}
            >>> net.boundary = (1,1)
            >>> net.neighbors(2, size=3)
            {1, 2, 3}
            >>> net.neighbors(0, 3)
            {0, 1, -1}

        .. rubric:: Erroneous Usage:

        .. doctest:: automata

            >>> net = ECA(30,boundary=(1, 1))
            >>> net.neighbors(5, 3)
            Traceback (most recent call last):
                ...
            ValueError: index must be a non-negative integer less than size
        """
        # Outgoing neighbors are a subset of incoming neighbors.
        return self.neighbors_in(index, size)

    def to_networkx_graph(self, size):
        """
        Return networkx graph given neet network. Requires networkx.

        :param size: size of ECA, required if network is an ECA
        :returns: a ``networkx.DiGraph``
        """

        edges = []
        for i in range(size):
            for j in self.neighbors_out(i, size):
                edges.append((i, j))

        return nx.DiGraph(edges, code=self.code, size=size,
                          boundary=self.boundary)

    def draw(self, size, filename=None):
        """
        Output a file with a simple network drawing.

        Requires networkx and pygraphviz.

        Supported image formats are determined by graphviz.  In particular,
        pdf support requires 'cairo' and 'pango' to be installed prior to
        graphviz installation.

        :param filename: filename to write drawing to. Temporary filename will
                         be used if no filename provided.
        :param size: size of ECA, required if network is an ECA
        :returns: a ``pygraphviz`` network drawing
        """
        nx.nx_agraph.view_pygraphviz(
            self.to_networkx_graph(size), prog='circo', path=filename)
