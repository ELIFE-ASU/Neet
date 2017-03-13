# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
from .landscape import StateSpace

class WTNetwork(object):
    def __init__(self, n):
        """
        Construct a weight/threshold boolean network with ``n`` nodes.

        .. rubric:: Example:

        ::

            >>> net = WTNetwork(5)
            >>> net.size
            5
            >>> WTNetwork(0)
            Traceback (most recent call last):
                ...
            ValueError: network must have at least one node

        :param n: the number of boolean nodes in the network
        :type n: int
        :raise TypeError: if ``n`` is not an ``int``
        :raise ValueError: if ``n < 1``
        """
        if not isinstance(n, int):
            raise(TypeError("n must be an int"))
        if n < 1:
            raise(ValueError("network must have at least one node"))

        self.__size = n

    @property
    def size(self):
        """
        The number of nodes in the network.

        .. rubric:: Example:

        ::

            >>> net = WTNetwork(5)
            >>> net.size
            5
            >>> WTNetwork(0)
            Traceback (most recent call last):
                ...
            ValueError: network must have at least one node

        :type: int
        """
        return self.__size

    def state_space(self):
        """
        Return a :class:`StateSpace` object for the network.

        ::

            >>> net = WTNetwork(3)
            >>> net.state_space()
            <neet.landscape.StateSpace object at 0x00000193E4DA84A8>
            >>> space = net.state_space()
            >>> list(space.states())
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :param n: the number of nodes in the lattice
        :type n: int
        :raises ValueError: if ``n < 1``
        """
        return StateSpace(self.size, b=2)

    def update(self, lattice):
        pass
