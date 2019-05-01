"""
Elementary Cellular Automata
============================

The :class:`neet.automata.eca.ECA` class describes an `Elementary Cellular
Automaton <https://en.wikipedia.org/wiki/Elementary_cellular_automaton>`_
with an arbitrary rule.

.. rubric:: Examples
"""
import numpy as np
from .network import BooleanNetwork


class ECA(BooleanNetwork):
    """
    ECA is a class to represent elementary cellular automaton rules. Each ECA
    contains an 8-bit integral member variable ``code`` representing the
    Wolfram code for the ECA rule and a set of boundary conditions which is
    either ``None``, signifying periodic boundary conditions, or a pair of
    cell states signifying fixed, open boundary conditions.
    """

    def __init__(self, code, size, boundary=None):
        """
        Construct an elementary cellular automaton rule.

        .. rubric:: Examples

        .. doctest:: automata

            >>> ca = ECA(30, 5)
            >>> ca.code
            30
            >>> ca.size
            5
            >>> ca.boundary
            >>> ca = ECA(30, 5, boundary=(0,0))
            >>> ca.boundary
            (0, 0)

        :param code: the Wolfram code for the ECA
        :type code: int
        :param size: the size of the ECA's lattice
        :type size: int
        :param boundary: the boundary conditions for the CA
        :type boundary: tuple or None
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\\{0,1,\\ldots,255\\}`
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of binary states
        """
        super(ECA, self).__init__(size)
        self.code = code
        self.boundary = boundary

    @property
    def code(self):
        """
        The Wolfram code of the elementary cellular automaton

        .. rubric:: Examples

        .. doctest:: automata

            >>> eca = ECA(30, 5)
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
        :raises ValueError: if ``code`` is not in :math:`\\{0,1,\\ldots,255\\}`
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
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        if not isinstance(size, int):
            raise TypeError("ECA size is not an int")
        if size < 1:
            raise ValueError("ECA size is negative")
        self._size = size
        self._volume = 2**size
        self._shape = [2] * size

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
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of binary states
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

    def neighbors_in(self, index, *args, **kwargs):
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
        if not isinstance(index, int):
            raise TypeError("index must be a non-negative integer")

        size = self.size

        if index < 0 or index > size - 1:
            msg = "index must be a non-negative integer less than size"
            raise ValueError(msg)

        left, right = index - 1, index + 1

        if left < 0 and self.boundary is None:
            left = size - 1

        if right > size - 1 and self.boundary is None:
            right = 0

        return {left, index, right}

    def neighbors_out(self, index, *args, **kwargs):
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

        size = self.size

        if index < 0 or index > size - 1:
            msg = "index must be a non-negative integer less than size"
            raise ValueError(msg)

        left, right = index - 1, index + 1

        if left < 0:
            left = size - 1 if self.boundary is None else 0

        if right > size - 1:
            right = 0 if self.boundary is None else size - 1

        return {left, index, right}

    def to_networkx_graph(self, *args, **kwargs):
        kwargs['code'] = self.code
        kwargs['boundary'] = self.boundary
        return super(ECA, self).to_networkx_graph(*args, **kwargs)


BooleanNetwork.register(ECA)
