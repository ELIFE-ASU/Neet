# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import re
from neet.statespace import StateSpace

class WTNetwork(object):
    """
    The WTNetwork class represents weight/threshold-based boolean networks. As
    such it is specified in terms of a matrix of edge weights and a vector of
    node thresholds, and each node of the network is expected to be in either
    of two states ``0`` or ``1``.
    """
    def __init__(self, weights, thresholds=None, names=None, theta=None):
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

        ::

            >>> net = WTNetwork(3)
            >>> net.size
            3
            >>> net.weights
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]])
            >>> net.thresholds
            array([ 0.,  0.,  0.])

        :param weights: the network weights
        :param thresholds: the network thresholds
        :param names: the names of the network nodes (optional)
        :parma theta: the threshold function to use
        :raises ValueError: if ``weights`` is empty
        :raises ValueError: if ``weights`` is not a square matrix
        :raises ValueError: if ``thresholds`` is not a vector
        :raises ValueError: if ``weights`` and ``thresholds`` have different dimensions
        :raises ValueError: if ``len(names)`` is not equal to the number of nodes
        :raises TypeError: if ``threshold_func`` is not callable
        """
        if type(weights) == int:
            self.weights = np.zeros([weights,weights])
        else:
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

        if theta is None:
            self.theta = type(self).split_threshold
        elif callable(theta):
            self.theta = theta
        else:
            raise(TypeError("theta must be a function"))

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
            ValueError: invalid network size

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

    def _unsafe_update(self, states, index=None, pin=None):
        """
        Update ``states``, in place, according to the network update rules
        without checking the validity of the arguments.

        .. rubric:: Basic Use:

        ::

            >>> net = WTNetwork.read("fission-net-nodes.txt", "fission-net-edges.txt")
            >>> net.size
            9
            >>> xs = [0,0,0,0,1,0,0,0,0]
            >>> net._unsafe_update(xs)
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
            >>> net._unsafe_update(xs)
            [0, 1, 1, 1, 0, 0, 1, 0, 0]

        .. rubric:: Single-Node Update:

        ::

            >>> xs = [0,0,0,0,1,0,0,0,0]
            >>> net._unsafe_update(xs, index=-1)
            [0, 0, 0, 0, 1, 0, 0, 0, 1]
            >>> net._unsafe_update(xs, index=2)
            [0, 0, 1, 0, 1, 0, 0, 0, 1]
            >>> net._unsafe_update(xs, index=3)
            [0, 0, 1, 1, 1, 0, 0, 0, 1]


        .. rubric:: State Pinning:

        ::

            >>> net._unsafe_update([0,0,0,0,1,0,0,0,0], pin=[-1])
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
            >>> net._unsafe_update([0,0,0,0,0,0,0,0,1], pin=[1])
            [0, 0, 1, 1, 0, 0, 1, 0, 0]
            >>> net._unsafe_update([0,0,0,0,0,0,0,0,1], pin=range(1,4))
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
            >>> net._unsafe_update([0,0,0,0,0,0,0,0,1], pin=[1,2,3,-1])
            [0, 0, 0, 0, 0, 0, 1, 0, 1]

        .. rubric:: Erroneous Usage:

        ::

            >>> net._unsafe_update([0,0,0])
            Traceback (most recent call last):
                ...
            ValueError: shapes (9,9) and (3,) not aligned: 9 (dim 1) != 3 (dim 0)
            >>> net._unsafe_update([0,0,0,0,2,0,0,0,0])
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
            >>> net._unsafe_update([0,0,0,0,1,0,0,0,0], 9)
            Traceback (most recent call last):
                ...
            IndexError: index 9 is out of bounds for axis 0 with size 9
            >>> net._unsafe_update([0,0,0,0,0,0,0,0,1], pin=[10])
            Traceback (most recent call last):
                ...
            IndexError: index 10 is out of bounds for axis 1 with size 9

        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :param index: the index to update or None
        :type index: int or None
        :param pin: the indices to pin (fix to their current state) or None
        :type pin: sequence
        :returns: the updated states
        """
        pin_states = pin is not None and pin != []
        if index is None:
            if pin_states:
                pinned = np.asarray(states)[pin]
            temp = np.dot(self.weights, states) - self.thresholds
            self.theta(temp, states)
            if pin_states:
                for (j,i) in enumerate(pin):
                    states[i] = pinned[j]
            return states
        else:
            temp = np.dot(self.weights[index], states) - self.thresholds[index]
            states[index] = self.theta(temp, states[index])
            return states


    def update(self, states, index=None, pin=None):
        """
        Update ``states``, in place, according to the network update rules.

        .. rubric:: Basic Use:

        ::

            >>> net = WTNetwork.read("fission-net-nodes.txt", "fission-net-edges.txt")
            >>> net.size
            9
            >>> xs = [0,0,0,0,1,0,0,0,0]
            >>> net.update(xs)
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
            >>> net.update(xs)
            [0, 1, 1, 1, 0, 0, 1, 0, 0]

        .. rubric:: Single-Node Update:

        ::

            >>> xs = [0,0,0,0,1,0,0,0,0]
            >>> net.update(xs, index=-1)
            [0, 0, 0, 0, 1, 0, 0, 0, 1]
            >>> net.(xs, index=2)
            [0, 0, 1, 0, 1, 0, 0, 0, 1]
            >>> net.(xs, index=3)
            [0, 0, 1, 1, 1, 0, 0, 0, 1]

        .. rubric:: State Pinning:

        ::

            >>> net.update([0,0,0,0,1,0,0,0,0], pin=[-1])
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
            >>> net.update([0,0,0,0,0,0,0,0,1], pin=[1])
            [0, 0, 1, 1, 0, 0, 1, 0, 0]
            >>> net.update([0,0,0,0,0,0,0,0,1], pin=range(1,4))
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
            >>> net.update([0,0,0,0,0,0,0,0,1], pin=[1,2,3,-1])
            [0, 0, 0, 0, 0, 0, 1, 0, 1]

        .. rubric:: Erroneous Usage:

        ::

            >>> net.update([0,0,0])
            Traceback (most recent call last):
                ...
            ValueError: incorrect number of states in array
            >>> net.update([0,0,0,0,2,0,0,0,0])
            Traceback (most recent call last):
                ...
            ValueError: invalid node state in states
            >>> net.update([0,0,0,0,1,0,0,0,0], 9)
            Traceback (most recent call last):
                ...
            IndexError: index 9 is out of bounds for axis 0 with size 9
            >>> net.update([0,0,0,0,1,0,0,0,0], index=-1, pin=[-1])
            Traceback (most recent call last):
                ...
            ValueError: cannot provide both the index and pin arguments
            >>> net.update([0,0,0,0,1,0,0,0,0], pin=[10])
            Traceback (most recent call last):
                ...
            IndexError: index 10 is out of bounds for axis 1 with size 9

        :param states: the one-dimensional sequence of node states
        :type states: sequence
        :param index: the index to update (or None)
        :type index: int or None
        :param pin: the indices to pin (or None)
        :type pin: sequence
        :returns: the updated states
        :raises TypeError: if ``states`` is not iterable
        :raises ValueError: if ``len(states)`` is not the number of nodes in the network
        :raises ValueError: if ``states[i] not in [0,1]`` for any node ``i``
        :raises IndexError: if ``index is not None and index > len(states)``
        :raises ValueError: if both ``index`` and ``pin`` are provided
        :raises IndexError: if any element of ``pin`` is greater than ``len(states)``
        """
        self.check_states(states)

        if (index is not None) and (pin is not None and pin != []):
            raise(ValueError("cannot provide both the index and pin arguments"))

        return self._unsafe_update(states, index, pin)

    @staticmethod
    def read(nodes_file, edges_file):
        """
        Read a network from a pair of node/edge files.

        Note that the node names cannot have spaces in them.

        .. rubric:: Examples:

        ::

            >>> net = WTNetwork.read("fission-net-nodes.txt", "fission-net-edges.txt")
            >>> net.size
            9
            >>> net.names
            ['SK', 'Cdc2_Cdc13', 'Ste9', 'Rum1', 'Slp1', 'Cdc2_Cdc13_active', 'Wee1_Mik1', 'Cdc25', 'PP']

        :returns: a :class:WTNetwork
        """
        comment = re.compile(r'^\s*#.*$')
        names, thresholds = [], []
        nameindices, index = dict(), 0
        with open(nodes_file, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    name, threshold = line.strip().split()
                    names.append(name)
                    nameindices[name] = index
                    thresholds.append(float(threshold))
                    index += 1

        n = len(names)
        weights = np.zeros((n,n), dtype=np.float)
        with open(edges_file, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    a, b, w = line.strip().split()
                    weights[nameindices[b],nameindices[a]] = float(w)

        return WTNetwork(weights, thresholds, names)

    @staticmethod
    def split_threshold(values, states):
        """
        Applies the following functional form to the arguments:

        .. math::

            \\theta_s(x,y) = \\begin{cases}
                0 & x < 0 \\\\
                y & x = 0 \\\\
                1 & x > 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y) in zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples:

        ::

            >>> xs = [0,0,0]
            >>> WTNetwork.split_threshold([1, -1, 0], xs)
            [1, 0, 0]
            >>> xs
            [1, 0, 0]
            >>> xs = [1,1,1]
            >>> WTNetwork.split_threshold([1, -1, 0], xs)
            [1, 0, 1]
            >>> xs
            [1, 0, 1]
            >>> WTNetwork.split_threshold(0,0)
            0
            >>> WTNetwork.split_threshold(0,1)
            1
            >>> WTNetwork.split_threshold(1,0)
            1
            >>> WTNetwork.split_threshold(1,1)
            1

        :param values: the threshold-shifted values of each node
        :param states: the pre-updated states of the nodes
        :returns: the updated states
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            for i, x in enumerate(values):
                if x < 0:
                    states[i] = 0
                elif x > 0:
                    states[i] = 1
            return states
        else:
            if values < 0:
                return 0
            elif values > 0:
                return 1
            return states

    @staticmethod
    def negative_threshold(values, states):
        """
        Applies the following functional form to the arguments:

        .. math::

            \\theta_n(x) = \\begin{cases}
                0 & x \\leq 0 \\\\
                1 & x > 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y) in zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples:

        ::

            >>> xs = [0,0,0]
            >>> WTNetwork.negative_threshold([1, -1, 0], xs)
            [1, 0, 0]
            >>> xs
            [1, 0, 0]
            >>> xs = [1,1,1]
            >>> WTNetwork.negative_threshold([1, -1, 0], xs)
            [1, 0, 0]
            >>> xs
            [1, 0, 0]
            >>> WTNetwork.negative_threshold(0,0)
            0
            >>> WTNetwork.negative_threshold(0,1)
            0
            >>> WTNetwork.negative_threshold(1,0)
            1
            >>> WTNetwork.negative_threshold(1,1)
            1

        :param values: the threshold-shifted values of each node
        :param states: the pre-updated states of the nodes
        :returns: the updated states
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            for i, x in enumerate(values):
                if x <= 0:
                    states[i] = 0
                else:
                    states[i] = 1
            return states
        else:
            if values <= 0:
                return 0
            else:
                return 1

    @staticmethod
    def positive_threshold(values, states):
        """
        Applies the following functional form to the arguments:

        .. math::

            \\theta_p(x) = \\begin{cases}
                0 & x < 0 \\\\
                1 & x \\geq 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y) in zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples:

        ::

            >>> xs = [0,0,0]
            >>> WTNetwork.positive_threshold([1, -1, 0], xs)
            [1, 0, 1]
            >>> xs
            [1, 0, 1]
            >>> xs = [1,1,1]
            >>> WTNetwork.positive_threshold([1, -1, 0], xs)
            [1, 0, 1]
            >>> xs
            [1, 0, 1]
            >>> WTNetwork.negative_threshold(0,0)
            1
            >>> WTNetwork.negative_threshold(0,1)
            1
            >>> WTNetwork.negative_threshold(1,0)
            1
            >>> WTNetwork.negative_threshold(-1,0)
            0

        :param values: the threshold-shifted values of each node
        :param states: the pre-updated states of the nodes
        :returns: the updated states
        """
        if isinstance(values, list) or isinstance(values, np.ndarray):
            for i, x in enumerate(values):
                if x < 0:
                    states[i] = 0
                else:
                    states[i] = 1
            return states
        else:
            if values < 0:
                return 0
            else:
                return 1
