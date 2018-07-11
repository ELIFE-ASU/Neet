# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
import networkx as nx
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
        if isinstance(weights, int):
            self.weights = np.zeros([weights, weights])
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

        self.metadata = {}

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
            <neet.statespace.StateSpace object at 0x00000193E4DA84A8>
            >>> space = net.state_space()
            >>> list(space)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :param n: the number of nodes in the lattice
        :type n: int
        :raises ValueError: if ``n < 1``
        """
        return StateSpace(self.size, base=2)

    def _unsafe_update(self, states, index=None, pin=None, values=None):
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

        .. rubric:: Value Fixing:

            >>> net.update([0,0,0,0,1,0,0,0,0], values={0:1, 2:1})
            [1, 0, 1, 0, 0, 0, 0, 0, 1]
            >>> net.update([0,0,0,0,0,0,0,0,1], values={0:1, 1:0, 2:0})
            [1, 0, 0, 1, 0, 0, 1, 0, 0]
            >>> net.update([0,0,0,0,0,0,0,0,1], values={-1:1, -2:1})
            [0, 1, 1, 1, 0, 0, 1, 1, 1]

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
        :param index: the index to update or None
        :param pin: the indices to pin (fix to their current state) or None
        :param values: a dictionary of index-value pairs to fix after update
        :returns: the updated states
        """
        pin_states = pin is not None and pin != []
        if index is None:
            if pin_states:
                pinned = np.asarray(states)[pin]
            temp = np.dot(self.weights, states) - self.thresholds
            self.theta(temp, states)
            if pin_states:
                for (j, i) in enumerate(pin):
                    states[i] = pinned[j]
        else:
            temp = np.dot(self.weights[index], states) - self.thresholds[index]
            states[index] = self.theta(temp, states[index])
        if values is not None:
            for key in values:
                states[key] = values[key]
        return states

    def update(self, states, index=None, pin=None, values=None):
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

        .. rubric:: Value Fixing:

            >>> net.update([0,0,0,0,1,0,0,0,0], values={0:1, 2:1})
            [1, 0, 1, 0, 0, 0, 0, 0, 1]
            >>> net.update([0,0,0,0,0,0,0,0,1], values={0:1, 1:0, 2:0})
            [1, 0, 0, 1, 0, 0, 1, 0, 0]
            >>> net.update([0,0,0,0,0,0,0,0,1], values={-1:1, -2:1})
            [0, 1, 1, 1, 0, 0, 1, 1, 1]

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
            >>> net.update([0,0,0,0,1,0,0,0,0], index=1, values={1:0,3:0,2:1})
            Traceback (most recent call last):
                ...
            ValueError: cannot provide both the index and values arguments
            >>> net.update([0,0,0,0,1,0,0,0,0], pin=[1], values={1:0,3:0,2:1})
            Traceback (most recent call last):
                ...
            ValueError: cannot set a value for a pinned state
            >>> net.update([0,0,0,0,1,0,0,0,0], values={1:2})
            Traceback (most recent call last):
                ...
            ValueError: invalid state in values argument

        :param states: the one-dimensional sequence of node states
        :param index: the index to update (or None)
        :param pin: the indices to pin (or None)
        :param values: a dictionary of index-value pairs to set after update
        :returns: the updated states
        :raises ValueError: if ``states`` is not in the network's state space
        :raises ValueError: if ``index`` and ``pin`` are both provided
        :raises ValueError: if ``index`` and ``values`` are both provided
        :raises ValueError: if an element of ``pin`` is a key in ``values``
        :raises ValueError: if a value in ``values`` is not binary (0 or 1)
        """
        if states not in self.state_space():
            raise ValueError("the provided state is not in the network's state space")

        if index is not None:
            if pin is not None and pin != []:
                raise ValueError("cannot provide both the index and pin arguments")
            elif values is not None and values != {}:
                raise ValueError("cannot provide both the index and values arguments")
        elif pin is not None and values is not None:
            for k in values.keys():
                if k in pin:
                    raise ValueError("cannot set a value for a pinned state")
        if values is not None:
            for val in values.values():
                if val != 0 and val != 1:
                    raise ValueError("invalid state in values argument")

        return self._unsafe_update(states, index, pin, values)

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
        weights = np.zeros((n, n), dtype=np.float)
        with open(edges_file, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    a, b, w = line.strip().split()
                    weights[nameindices[b], nameindices[a]] = float(w)

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

            >>> ys = [0,0,0]
            >>> WTNetwork.split_threshold([1, -1, 0], ys)
            [1, 0, 0]
            >>> ys
            [1, 0, 0]
            >>> ys = [1,1,1]
            >>> WTNetwork.split_threshold([1, -1, 0], ys)
            [1, 0, 1]
            >>> ys
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

            >>> ys = [0,0,0]
            >>> WTNetwork.negative_threshold([1, -1, 0], ys)
            [1, 0, 0]
            >>> ys
            [1, 0, 0]
            >>> ys = [1,1,1]
            >>> WTNetwork.negative_threshold([1, -1, 0], ys)
            [1, 0, 0]
            >>> ys
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

            >>> ys = [0,0,0]
            >>> WTNetwork.positive_threshold([1, -1, 0], ys)
            [1, 0, 1]
            >>> ys
            [1, 0, 1]
            >>> ys = [1,1,1]
            >>> WTNetwork.positive_threshold([1, -1, 0], ys)
            [1, 0, 1]
            >>> ys
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

    def neighbors_in(self, index):
        """
        Return the set of all neighbor nodes, where
        edge(neighbor_node-->index) exists.

        :param index: node index
        :returns: the set of all node indices which point toward the index node

        .. rubric:: Basic Use:

        ::

            >>> net = WTNetwork(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [ 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0,-1.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0],
             [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,-1.0]],
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
            >>> net.neighbors_in(2)
            set([0, 1, 5, 8])
        """
        if (self.theta is type(self).negative_threshold) or (self.theta is type(self).positive_threshold):
            return set(np.flatnonzero(self.weights[index]))
        else: 
            ## Assume every other theta has self loops. This will be depreciated
            ## when we convert all WTNetworks to logicnetworks by default.
            return set(np.flatnonzero(self.weights[index]))|set([index])

    def neighbors_out(self, index):
        """
        Return the set of all neighbor nodes, where
        edge(index-->neighbor_node) exists.

        :param index: node index
        :returns: the set of all node indices which the index node points to

        .. rubric:: Basic Use:

        ::

            >>> net = WTNetwork(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [ 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0,-1.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0],
             [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,-1.0]],
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
            >>> net.neighbors_out(2)
            set([1, 5])
        """
        if (self.theta is type(self).negative_threshold) or (self.theta is type(self).positive_threshold):
            return set(np.flatnonzero(self.weights[:, index]))

        else: 
            ## Assume every other theta has self loops. This will be depreciated
            ## when we convert all WTNetworks to logicnetworks by default.
            return set(np.flatnonzero(self.weights[:, index]))|set([index])

    def neighbors(self, index):
        """
        Return a set of neighbors for a specified node, or a list of sets of
        neighbors for all nodes in the network.

        :param index: node index
        :returns: a set (if index!=None) or list of sets of neighbors of a node or network or nodes

        .. rubric:: Basic Use:

        ::

            >>> net = WTNetwork(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [-1.0,-1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 1.0],
             [ 0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.0, 0.0, 0.0],
             [ 0.0, 0.0,-1.0,-1.0,-1.0, 0.0,-1.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
             [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-1.0],
             [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,-1.0]],
            [ 0.0,-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
            >>> net.neighbors(2)
            set([0, 1, 5, 8])
        """
        return self.neighbors_in(index) | self.neighbors_out(index)

    def to_networkx_graph(self,labels='indices'):
        """
        Return networkx graph given neet network.  Requires networkx.

        :param labels: how node is labeled and thus identified in networkx graph 
                       ('names' or 'indices')
        :returns: a networkx DiGraph
        """
        if labels == 'names':
            if hasattr(self,'names') and (self.names != None):
                labels = self.names
            else:
                raise ValueError("network nodes do not have names")

        elif labels == 'indices':
            labels = range(self.size)

        else:
            raise ValueError("labels must be 'names' or 'indices'")

        edges = []
        for i,label in enumerate(labels):
            for j in self.neighbors_out(i):
                edges.append((labels[i],labels[j]))

        return nx.DiGraph(edges,name=self.metadata.get('name'))

    def draw(self,labels='indices',filename=None):
        """
        Output a file with a simple network drawing.  
        
        Requires networkx and pygraphviz.
        
        Supported image formats are determined by graphviz.  In particular,
        pdf support requires 'cairo' and 'pango' to be installed prior to
        graphviz installation.

        :param labels: how node is labeled and thus identified in networkx graph 
                   ('names' or 'indices'), only used if network is a LogicNetwork or WTNetwork
        :param filename: filename to write drawing to. Temporary filename will be used if no filename provided.
        :returns: a pygraphviz network drawing
        
        """        
        nx.nx_agraph.view_pygraphviz(self.to_networkx_graph(labels=labels),prog='circo',path=filename)
