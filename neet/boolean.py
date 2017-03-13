# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import re
from .landscape import StateSpace

class WTNetwork(object):
    def __init__(self, weights, thresholds=None, names=None):
        """
        Construct a network from weights and thresholds.

        .. rubric:: Examples

        ::

            >>> net = WTNetwork([[1,0],[1,1]])
            >>> net.size
            2
            >>> net.weights
            array([[ 1.,  0.],
                   [ 1.,  1.]])
            >>> net.thresholds
            array([ 0.,  0.])

        ::

            >>> net = WTNetwork([[1,0],[1,1]], [0.5,-0.5])
            >>> net.size
            2
            >>> net.weights
            array([[ 1.,  0.],
                   [ 1.,  1.]])
            >>> net.thresholds
            array([ 0.5, -0.5])

        :param weights: the network weights
        :param thresholds: the network thresholds
        :param names: the names of the network nodes (optional)
        :raises ValueError: if ``weights`` is empty
        :raises ValueError: if ``weights`` is not a square matrix
        :raises ValueError: if ``thresholds`` is not a vector
        :raises ValueError: if ``weights`` and ``thresholds`` have different dimensions
        :raises ValueError: if ``len(names)`` is not equal to the number of nodes
        """
        self.weights = np.asarray(weights, dtype=np.float)
        shape = self.weights.shape
        if self.weights.ndim != 2:
            raise(ValueError("weights must be a matrix"))
        elif shape[0] != shape[1]:
            raise(ValueError("weights must be square"))

        if thresholds is None:
            self.thresholds = np.zeros(shape[1], dtype=np.float)
        else:
            self.thresholds = np.asarray(thresholds, dtype=np.float)

        self.__size = self.thresholds.size

        if isinstance(names, str):
            self.names = list(names)
        else:
            self.names = names

        if self.thresholds.ndim != 1:
            raise(ValueError("thresholds must be a vector"))
        elif shape[0] != self.size:
            raise(ValueError("weights and thresholds have different dimensions"))
        elif self.size < 1:
            raise(ValueError("invalid network size"))
        elif names is not None and len(names) != self.size:
            raise(ValueError("either all or none of the nodes may have a name"))

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

    def _unsafe_update(self, states):
        """
        Update ``states``, in place, according to the network update rules
        without checking the validity of the arguments.

        .. rubric:: Examples:

        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :returns: the updated states
        """
        temp = np.dot(self.weights, states) - self.thresholds
        for (i,x) in enumerate(temp):
            if x < 0.0:
                states[i] = 0
            elif x > 0.0:
                states[i] = 1
        return states

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

    @staticmethod
    def read(nodes_file, edges_file):
        """
        Read a network from a pair of node/edge files.

        .. rubric:: Examples:

        :returns: a :class:WTNetwork
        """
        comment = re.compile(r'^\s*#.*$')
        names = dict()
        thresholds = []
        index = 0
        with open(nodes_file, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    name, threshold = line.strip().split()
                    names[name] = index
                    thresholds.append(float(threshold))
                    index += 1

        n = len(names)
        weights = np.empty((n,n), dtype=np.float)
        with open(edges_file, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    a, b, w = line.strip().split()
                    weights[names[b],names[a]] = float(w)

        net = WTNetwork(n)
        net._WTNetwork__thresholds = np.asarray(thresholds)
        net._WTNetwork__weights    = weights
        return net, names
