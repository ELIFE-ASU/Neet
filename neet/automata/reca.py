# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
from neet.landscape import StateSpace
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

        .. rubric:: Examples:

        ::

            >>> reca = RewiredECA(30, size=3)
            >>> reca.code
            30
            >>> reca.size
            3
            >>> reca.wiring
            array([[-1,  0,  1],
                   [ 0,  1,  2],
                   [ 1,  2,  0]])

        ::

            >>> reca = RewiredECA(30, wiring=[[0,1,2],[-1,0,0],[2,2,1]])
            >>> reca.code
            30
            >>> reca.size
            3
            >>> reca.wiring
            array([[ 0,  1,  2],
                   [-1,  0,  0],
                   [ 2,  2,  1]])

        :param code: the 8-bit Wolfram code for the rule
        :type code: int
        :param boundary: the boundary conditions for the CA
        :type boundary: tuple or None
        :param size: the number of cells in the lattice
        :type size: int or None
        :param wiring: a wiring matrix
        :raises ValueError: if ``size is None and wiring is None``
        :raises ValueError: if ``size is not None and wiring is not None``
        :raises TypeError: if ``size is not None and not isinstance(size, int)``
        :raises ValueError: if ``size is not None and size <= 0``
        :raises TypeError: if ``not isinstance(wiring, list) and not
            isinstance(wiring, numpy.ndarray)``
        :raises ValueError: if ``wiring`` is not :math:`3 \times N`
        :raises ValueError: if ``any(wiring < -1) or any(wiring >= N)``
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
                self.__wiring[0, :] = range(-1, size-1)
                self.__wiring[1, :] = range(0, size)
                self.__wiring[2, :-1] = range(1, size)
        elif wiring is not None:
            if not isinstance(wiring, list) and not isinstance(wiring, np.ndarray):
                raise TypeError("wiring must be a list or an array")
            wiring_array = np.copy(wiring)
            shape = wiring_array.shape
            if wiring_array.ndim != 2:
                raise ValueError("wiring must be a matrix")
            elif shape[0] != 3:
                raise ValueError("wiring must have 3 rows")
            elif np.any(wiring_array < -1):
                raise ValueError("invalid input node in wiring")
            elif np.any(wiring_array >= shape[1]):
                raise ValueError("invalid input node in wiring")
            self.__size = shape[1]
            self.__wiring = wiring_array
        else:
            raise ValueError("either size or wiring must be provided")

    @property
    def wiring(self):
        """
        The wiring matrix for the rule.

        .. rubric:: Examples:

        ::

            >>> eca = RewiredECA(30, size=3)
            >>> eca.wiring
            array([[-1,  0,  1],
                   [ 0,  1,  2],
                   [ 1,  2,  0]])
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

        .. rubric:: Examples:

        ::

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
        Return a :class:`StateSpace` object for the cellular automaton lattice.

        .. rubric:: Examples:

        ::

            >>> eca = RewiredECA(30, size=3)
            >>> eca.state_space()
            <neet.landscape.StateSpace object at 0x0000020EED289748>
            >>> space = eca.state_space()
            >>> list(space.states())
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :returns: :class:`StateSpace`
        """
        return StateSpace(self.__size, b=2)
