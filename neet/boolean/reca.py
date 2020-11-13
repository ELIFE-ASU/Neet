"""
.. currentmodule:: neet.boolean

.. testsetup:: reca

    from neet.boolean import RewiredECA
"""
import numpy as np
from .network import BooleanNetwork


class RewiredECA(BooleanNetwork):
    """
    RewiredECA represents elementary cellular automaton rule with a rewired
    topology. That is, RewiredECA is a variant of an :class:`neet.boolean.ECA`
    wherein the neighbors of a given cell can be specified by the user. This
    allows one to study, for example, the role of topology in the dynamics of a
    network. Every :class:`neet.boolean.ECA` can be represented as a RewiredECA
    with standard wiring, but all RewiredECA are *fixed sized* networks. For
    this reason, RewiredECA **does not** derive from :class:`neet.boolean.ECA`.

    .. inheritance-diagram:: RewiredECA
        :parts: 1

    RewiredECA instances can be instantiated by providing an ECA rule ``code``,
    and either the number of nodes in the network (``size``) or a ``wiring``
    matrix which specifies how the nodes are wired.  Optionally, the user can
    specify boundary conditions as in :class:`neet.boolean.ECA`. As with all
    :class:`neet.Network` classes, the names of the nodes and network-wide
    metadata can be provided.

    In addition to all inherited methods, RewiredECA exposes the following properites

    .. autosummary::
        :nosignatures:

        code
        boundary
        wiring

    .. rubric:: Examples

    If ``wiring`` is not provided, the network is wired as a standard
    :class:`neet.boolean.ECA`.

    .. doctest:: reca

        >>> reca = RewiredECA(30, size=5)
        >>> reca.code
        30
        >>> reca.size
        5
        >>> reca.wiring
        array([[-1,  0,  1,  2,  3],
               [ 0,  1,  2,  3,  4],
               [ 1,  2,  3,  4,  5]])

    Wiring matrices are :math:`3 \times N` matrices where each column is a node
    of the network, and the rows represent the left-, middle- and right-input
    for the nodes. The number of nodes will be inferred from the width of the
    matrix. For example:

    .. doctest:: reca

        >>> reca = RewiredECA(30, wiring=[[0,1,2],[-1,0,0],[2,3,1]])
        >>> reca.code
        30
        >>> reca.size
        3
        >>> reca.wiring
        array([[ 0,  1,  2],
               [-1,  0,  0],
               [ 2,  3,  1]])

    Here the :math:`0`th node takes input from nodes :math:`0`, :math:`-1` and
    :math:`2` as left, middle and right input. Note that ``-1`` represents the
    left-boundary condition of the RewiredECA. If instance has periodic
    boundary conditions then ``-1`` is effectively ``N-1``. Similarly ``N`` is
    the right boundary condition.

    To see how the wiring affects the result:

    .. doctest:: reca

        >>> ca = RewiredECA(30, size=3)
        >>> ca.update([0, 1, 0])
        [1, 1, 1]
        >>> ca = RewiredECA(30, wiring=[[0,1,3], [1,1,1], [2,1,2]])
        >>> ca.update([0, 1, 0])
        [1, 0, 1]

    :param code: the 8-bit Wolfram code for the rule
    :type code: int
    :param boundary: the boundary conditions for the CA
    :type boundary: tuple, None
    :param size: the number of cells in the lattice
    :type size: int or None
    :param wiring: a wiring matrix
    :type wiring: list, numpy.ndarray
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    :raises ValueError: if both ``size`` and ``wiring`` are provided
    :raises ValueError: if neither ``size`` nor ``wiring`` are provided
    :raises ValueError: if ``size`` is less than :math:`1` (when provided)
    :raises ValueError: if ``wiring`` is not a :math:`3 \\times N` matrix (when
                        provided)
    :raises ValueError: if any element of ``wiring`` is outside the range
                        :math:`[-1, ``size``]` (when provided)
    """

    def __init__(self, code, boundary=None, size=None, wiring=None, names=None, metadata=None):
        if size is not None and wiring is not None:
            raise ValueError("cannot provide size and wiring at the same time")
        elif size is not None:
            super(RewiredECA, self).__init__(size, names=names, metadata=metadata)
            self.code = code
            self.boundary = boundary
            self.__wiring = np.zeros((3, size), dtype=int)
            self.__wiring[0, :] = range(-1, size - 1)
            self.__wiring[1, :] = range(0, size)
            self.__wiring[2, :] = range(1, size + 1)
        elif wiring is not None:
            if not isinstance(wiring, (list, np.ndarray)):
                raise TypeError("wiring must be a list or an array")
            wiring_array = np.copy(wiring)
            shape = wiring_array.shape
            if wiring_array.ndim != 2:
                raise ValueError("wiring must be a matrix")
            elif shape[0] != 3:
                raise ValueError("wiring must have 3 rows")
            elif np.any(wiring_array < -1):
                raise ValueError("invalid input node in wiring")
            elif np.any(wiring_array > shape[1]):
                raise ValueError("invalid input node in wiring")

            super(RewiredECA, self).__init__(int(shape[1]), names=names, metadata=metadata)
            self.code = code
            self.boundary = boundary
            self.__wiring = wiring_array
        else:
            raise ValueError("either size or wiring must be provided")

    @property
    def code(self):
        """
        The Wolfram code of the elementary cellular automaton

        .. rubric:: Examples

        .. doctest:: reca

            >>> reca = RewiredECA(30, size=55)
            >>> reca.code
            30
            >>> reca.code = 45
            >>> reca.code
            45
            >>> reca.code = 256
            Traceback (most recent call last):
                ...
            ValueError: invalid ECA code

        :type: int
        :raises ValueError: if code is not in :math:`\\{0,1,\\ldots,255\\}`
        """
        return self.__code

    @code.setter
    def code(self, code):
        if not isinstance(code, int):
            raise TypeError("ECA code is not an int")
        if 255 < code or code < 0:
            raise ValueError("invalid ECA code")
        self.__code = code
        self.clear_landscape()

    @property
    def boundary(self):
        """
        The boundary conditions of the elemenary cellular automaton

        .. rubric:: Examples

        .. doctest:: reca

            >>> reca = RewiredECA(30, size=5)
            >>> reca.boundary
            >>> reca.boundary = (0,1)
            >>> reca.boundary
            (0, 1)
            >>> reca.boundary = None
            >>> reca.boundary
            >>> reca.boundary = [0,1]
            Traceback (most recent call last):
                ...
            TypeError: ECA boundary are neither None nor a tuple

        :type: tuple, None
        :raises ValueError: if boundary is neither None nor a pair of binary states
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

    @property
    def wiring(self):
        """
        The wiring matrix for the rule.

        .. rubric:: Examples

        .. doctest:: reca

            >>> reca = RewiredECA(30, size=4)
            >>> reca.wiring
            array([[-1,  0,  1,  2],
                   [ 0,  1,  2,  3],
                   [ 1,  2,  3,  4]])
            >>> eca = RewiredECA(30, wiring=[[0,1],[1,1],[-1,-1]])
            >>> eca.wiring
            array([[ 0,  1],
                   [ 1,  1],
                   [-1, -1]])

        :type: numpy.ndarray
        """
        return self.__wiring

    def _unsafe_update(self, lattice, index=None, pin=None, values=None):
        pin_states = pin is not None and pin != []
        if self.boundary:
            left = self.boundary[0]
            right = self.boundary[1]
        else:
            left = lattice[-1]
            right = lattice[0]

        code = self.code
        wiring = self.wiring
        size = len(lattice)

        if index is None:
            if pin_states:
                pinned = np.asarray(lattice)[pin]
            temp = np.copy(lattice)
            for j in range(size):
                shift = 0
                for i in range(3):
                    k = wiring[i, j]
                    if k == -1:
                        shift = 2 * shift + left
                    elif k == size:
                        shift = 2 * shift + right
                    else:
                        shift = 2 * shift + lattice[k]
                temp[j] = 1 & (code >> (7 & shift))
            lattice[:] = temp[:]
            if pin_states:
                for j, i in enumerate(pin):
                    lattice[i] = pinned[j]
        else:
            if index < 0:
                index += len(lattice)
            shift = 0
            for i in range(3):
                k = wiring[i, index]
                if k == -1:
                    shift = 2 * shift + left
                elif k == size:
                    shift = 2 * shift + right
                else:
                    shift = 2 * shift + lattice[k]
                lattice[index] = 1 & (code >> (7 & shift))

        if values is not None:
            for key in values:
                lattice[key] = values[key]

        return lattice

    def neighbors_in(self, index, *args, **kwargs):
        if not isinstance(index, int):
            raise TypeError("index must be a non-negative integer")

        if index < 0 or index > self.size - 1:
            return set()
        else:
            return set(self.wiring[:, index])

    def neighbors_out(self, index, *args, **kwargs):
        if not isinstance(index, int):
            raise TypeError("index must be a non-negative integer")

        neighbors = set()
        for j in range(self.size):
            for i in range(3):
                if self.wiring[i, j] == index:
                    neighbors.add(j)

        return neighbors

    def network_graph(self, *args, **kwargs):
        kwargs['code'] = self.code
        kwargs['boundary'] = self.boundary
        g = super(RewiredECA, self).network_graph(*args, **kwargs)

        if 'labels' not in kwargs:
            kwargs['labels'] = 'indices'

        if kwargs['labels'] == 'indices':
            g.add_edges_from(map(lambda n: (-1, n), self.neighbors_out(-1)))
            g.add_edges_from(map(lambda n: (5, n), self.neighbors_out(5)))
        elif kwargs['labels'] == 'names':
            names = self.names
            g.add_edges_from(map(lambda n: ('left', names[n]), self.neighbors_out(-1)))
            g.add_edges_from(map(lambda n: ('right', names[n]), self.neighbors_out(5)))

        return g


BooleanNetwork.register(RewiredECA)
