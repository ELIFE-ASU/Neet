"""
.. currentmodule:: neet.boolean

.. testsetup:: eca

    from neet.boolean import ECA

The :class:`neet.boolean.ECA` class describes an `Elementary Cellular Automaton
<https://en.wikipedia.org/wiki/Elementary_cellular_automaton>`_ with an
arbitrary rule.
"""
import numpy as np
from . import BooleanNetwork


class ECA(BooleanNetwork):
    """
    ECA represents an elementary cellular automaton rule. Each ECA contains an
    8-bit integral member variable ``code`` representing the Wolfram code for
    the ECA rule and a set of boundary conditions which is either ``None``,
    signifying periodic boundary conditions, or a pair of cell states
    signifying fixed, open boundary conditions.  As with all
    :class:`neet.Network` classes, the names of the nodes and network-wide
    metadata can be provided.

    .. inheritance-diagram:: ECA
        :parts: 1

    In addition to all inherited methods, ECA exposes the following properites:

    .. autosummary::
        :nosignatures:

        code
        boundary

    :param code: the Wolfram code for the ECA
    :type code: int
    :param size: the size of the ECA's lattice
    :type size: int
    :param boundary: the boundary conditions for the CA
    :type boundary: tuple or None
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    :raises ValueError: if ``code`` is not in :math:`\\{0,1,\\ldots,255\\}`
    :raises ValueError: if ``boundary`` is a neither ``None`` nor a pair of binary states
    """

    def __init__(self, code, size, boundary=None, names=None, metadata=None):
        super(ECA, self).__init__(size, names=names, metadata=metadata)
        self.code = code
        self.boundary = boundary

    @property
    def code(self):
        """
        The Wolfram code of the elementary cellular automaton.

        .. rubric:: Examples

        .. doctest:: eca

            >>> eca = ECA(30, size=5)
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
        :raises ValueError: if ``code`` is not in :math:`\\{0,1,\\ldots,255\\}`
        """
        return self.__code

    @code.setter
    def code(self, code):
        if not isinstance(code, int):
            raise TypeError("ECA code is not an int")
        if 255 < code or code < 0:
            raise ValueError("invalid ECA code")
        self.clear_landscape()
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
        self.clear_landscape()

    @property
    def boundary(self):
        """
        The boundary conditions of the elemenary cellular automaton.

        .. rubric:: Examples

        .. doctest:: eca

            >>> eca = ECA(30, size=5)
            >>> eca.boundary
            >>> eca.boundary = (0, 1)
            >>> eca.boundary
            (0, 1)
            >>> eca.boundary = None
            >>> eca.boundary
            >>> eca.boundary = [0, 1]
            Traceback (most recent call last):
                ...
            TypeError: ECA boundary are neither None nor a tuple

        :type: tuple, None
        :raises ValueError: if ``boundary`` is a neither ``None`` nor a pair of binary states
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
        self.clear_landscape()

    def _unsafe_update(self, lattice, index=None, pin=None, values=None):
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

    def network_graph(self, *args, **kwargs):
        kwargs['code'] = self.code
        kwargs['boundary'] = self.boundary
        return super(ECA, self).network_graph(*args, **kwargs)


BooleanNetwork.register(ECA)
