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

        self.__size       = n
        self.__thresholds = np.zeros(n, dtype=np.int)
        self.__weights    = np.zeros((n,n), dtype=np.int)

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

    def check_states(self, states):
        """
        Check the validity of the provided states.

        .. rubric:: Examples;

        ::

            >>> WTNetwork(3).check_states([0,0,0])
            True
            >>> WTNetwork(3).check_states([0,0])
            Traceback (most recent call last):
                ...
            ValueError: incorrect number of states in array
            >>> WTNetwork(3).check_states([1,2,1])
                ...
            ValueError: invalid node state in states

        :returns: ``True`` if the ``states`` are valid, otherwise an error is raised
        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :raises TypeError: if ``states`` is not iterable
        :raises ValueError: if ``len(states)`` is not the number of nodes in the network
        :raises ValueError: if ``states[i] not in [0,1]`` for any node ``i``
        """
        if len(states) != self.size:
            raise(ValueError("incorrect number of states in array"))
        for x in states:
            if x !=0 and x != 1:
                raise(ValueError("invalid node state in states"))
        return True

    def _unsafe_update(self, lattice):
        """
        Update ``states``, in place, according to the network update rules
        without checking the validity of the arguments.

        .. rubric:: Examples:

        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :returns: the updated states
        """
        temp = np.dot(self.__weights, lattice) - self.__thresholds
        for (i,x) in enumerate(temp):
            if x < 0.0:
                lattice[i] = 0
            elif x > 0.0:
                lattice[i] = 1
        return lattice

    def update(self, states):
        """
        Update ``states``, in place, according to the network update rules.

        .. rubric:: Examples:

        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :returns: the updated states
        :raises TypeError: if ``states`` is not iterable
        :raises ValueError: if ``len(states)`` is not the number of nodes in the network
        :raises ValueError: if ``states[i] not in [0,1]`` for any node ``i``
        """
        self.check_states(states)
        return self._unsafe_update(states)
