"""
Rewired Elementary Cellular Automata
====================================

The :class:`neet.automata.reca.RewiredECA` implements a variant of an ECA
wherein the neighbors of a give cell can be specified by the user. This
allows one to study, for example, the role of topology in the dynamics of a
network. Every ``ECA`` can be represented as a ``RewiredECA`` with standard
wiring, but all ``RewiredECA`` are *fixed sized* networks.

.. rubric:: Examples

.. doctest:: automata

    >>> ca = RewiredECA(30, size=3)
    >>> ca.update([0, 1, 0])
    [1, 1, 1]
    >>> ca = RewiredECA(30, wiring=[[0,1,3], [1,1,1], [2,1,2]])
    >>> ca.update([0, 1, 0])
    [1, 0, 1]
"""
import numpy as np
from neet.statespace import StateSpace
from neet.interfaces import Network
from . import eca


class RewiredECA(eca.ECA):
    """
    RewiredECA is a class to represent elementary cellular automata rules with
    arbitrarily defined topology. Since the topology must be provided,
    RewiredECA are naturally fixed-sized.
    """

    def __init__(self, code, boundary=None, size=None, wiring=None):
        """
        Construct a rewired elementary cellular automaton rule.

        .. rubric:: Examples

        .. doctest:: automata

            >>> reca = RewiredECA(30, size=3)
            >>> reca.code
            30
            >>> reca.size
            3
            >>> reca.wiring
            array([[-1,  0,  1],
                   [ 0,  1,  2],
                   [ 1,  2,  3]])

        .. doctest:: automata

            >>> reca = RewiredECA(30, wiring=[[0,1,2],[-1,0,0],[2,3,1]])
            >>> reca.code
            30
            >>> reca.size
            3
            >>> reca.wiring
            array([[ 0,  1,  2],
                   [-1,  0,  0],
                   [ 2,  3,  1]])

        :param code: the 8-bit Wolfram code for the rule
        :type code: int
        :param boundary: the boundary conditions for the CA
        :type boundary: tuple or None
        :param size: the number of cells in the lattice
        :type size: int or None
        :param wiring: a wiring matrix
        :raises ValueError: if ``size is None and wiring is None``
        :raises ValueError: if ``size is not None and wiring is not None``
        :raises TypeError: if ``size is not None and not
                           isinstance(size, int)``
        :raises ValueError: if ``size is not None and size <= 0``
        :raises TypeError: if ``not isinstance(wiring, list) and not
                           isinstance(wiring, numpy.ndarray)``
        :raises ValueError: if ``wiring`` is not :math:`3 \times N`
        :raises ValueError: if ``any(wiring < -1) or any(wiring > N)``
        """
        super(RewiredECA, self).__init__(code, boundary=boundary)
        if size is not None and wiring is not None:
            raise ValueError("cannot provide size and wiring at the same time")
        elif size is not None:
            if not isinstance(size, int):
                raise TypeError("size must be an int")
            elif size <= 0:
                raise ValueError("size must be positive, nonzero")
            else:
                self.__size = size
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
            self.__size = int(shape[1])
            self.__wiring = wiring_array
        else:
            raise ValueError("either size or wiring must be provided")

    @property
    def wiring(self):
        """
        The wiring matrix for the rule.

        .. rubric:: Examples

        .. doctest:: automata

            >>> reca = RewiredECA(30, size=3)
            >>> reca.wiring
            array([[-1,  0,  1],
                   [ 0,  1,  2],
                   [ 1,  2,  3]])
            >>> eca = RewiredECA(30, wiring=[[0,1],[1,1],[-1,-1]])
            >>> eca.wiring
            array([[ 0,  1],
                   [ 1,  1],
                   [-1, -1]])

        :type: ``numpy.ndarray``
        """
        return self.__wiring

    @property
    def size(self):
        """
        The number of cells in the CA lattice.

        .. rubric:: Examples

        .. doctest:: automata

            >>> eca = RewiredECA(30, size=3)
            >>> eca.size
            3
            >>> eca = RewiredECA(30, wiring=[[-1,0], [0,1], [1,0]])
            >>> eca.size
            2

        :type: int
        """
        return self.__size

    def state_space(self):
        """
        Return a :class:`neet.statespace.StateSpace` object for the
        cellular automaton lattice.

        .. rubric:: Examples

        .. doctest:: automata

            >>> eca = RewiredECA(30, size=3)
            >>> eca.state_space()
            <neet.statespace.StateSpace object at 0x...>
            >>> space = eca.state_space()
            >>> list(space)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :returns: :class:`neet.statespace.StateSpace`
        """
        return StateSpace(self.__size, base=2)

    def _unsafe_update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice``, in place, without
        checking the validity of the arguments.

        :param lattice: the one-dimensional sequence of states
        :param index: the index to update (optional)
        :param pin: a sequence of indices to fix (optional)
        :param values: a dict of index/value pairs to set (optional)
        :returns: the updated lattice
        """
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

    def update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice`` in place.

        .. rubric:: Examples

        .. doctest:: automata

            >>> reca = RewiredECA(30, size=5)
            >>> reca.update([1,0,0,0,0])
            [1, 1, 0, 0, 1]
            >>> reca.wiring[:,:] = [[-1, 2, 1, 2, -1], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5]]
            >>> reca.update([0,0,1,0,0])
            [0, 0, 1, 1, 0]
            >>> reca.update([1,1,1,1,1])
            [0, 0, 0, 0, 0]
            >>> reca.update([1,1,1,1,1], index=2)
            [1, 1, 0, 1, 1]
            >>> reca.update([1,1,1,1,1], pin=[1, 3])
            [0, 1, 0, 1, 0]
            >>> reca.update([1,1,1,1,1], values={0: 1, -1: 1})
            [1, 0, 0, 0, 1]

        :param lattice: the one-dimensional sequence of states
        :param index: the index to update (optional)
        :param pin: a sequence of indices to fix (optional)
        :param values: a dict of index/value pairs to set (optional)
        :returns: the updated lattice
        """
        size = len(lattice)
        if lattice not in self.state_space():
            msg = "the provided state is not in the RewiredECA's state space"
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

        return self._unsafe_update(lattice, index=index, pin=pin, values=values)


Network.register(RewiredECA)
