# -*- coding: utf-8 -*-
"""
.. currentmodule:: neet.boolean

.. testsetup:: wtnetwork

    from neet.boolean import WTNetwork
"""
import numpy as np
import re
from .network import BooleanNetwork


class WTNetwork(BooleanNetwork):
    """
    WTNetwork represents weight-threshold boolean network. This type of Boolean
    network model is common in biology as it represents activating/inhibiting
    interactions between subcomponents.

    .. inheritance-diagram:: WTNetwork
        :parts: 1

    In addition to methods inherited from :class:`neet.boolean.BooleanNetwork`,
    WTNetwork exposes the following attributes

    +--------------------+----------------------------------------------------+
    | :attr:`weights`    | The network's square weight matrix.                |
    +--------------------+----------------------------------------------------+
    | :attr:`thresholds` | The network's threshold vector.                    |
    +--------------------+----------------------------------------------------+
    | :attr:`theta`      | The network's activation function.                 |
    +--------------------+----------------------------------------------------+

    and static methods:

    .. autosummary::
        :nosignatures:

        read
        positive_threshold
        negative_threshold
        split_threshold

    At a minimum, WTNetworks accept either a weight matrix or a size. The
    weight matrix must be square, with the :math:`(i,j)` element representing
    the weight on the edge from :math:`j`-th node to the :math:`i`-th. If a
    size is provided, all weights are assumed to be :math:`0.0`.

    .. doctest:: wtnetwork

        >>> WTNetwork(3)
        <neet.boolean.wtnetwork.WTNetwork object at 0x...>
        >>> WTNetwork([[0, 1, 0], [-1, 0, -1], [-1, 1, 1]])
        <neet.boolean.wtnetwork.WTNetwork object at 0x...>

    Each node has associated with it a threshold value. These thresholds can be
    provided at initialization. If none are provided, all thresholds are
    assumed to be :math:`0.0`.

    .. doctest:: wtnetwork

        >>> net = WTNetwork(3, [0.5, 0.0, -0.5])
        >>> net.thresholds
        array([ 0.5,  0. , -0.5])
        >>> WTNetwork([[0, 1, 0], [-1, 0, -1], [-1, 1, 1]], thresholds=[0.5, 0.0, -0.5])
        <neet.boolean.wtnetwork.WTNetwork object at 0x...>

    Finally, every node of the network is assumed to use the same activation
    function, ``theta``. This function, if not provided, is assumed to be
    :meth:`split_threshold`.

    .. doctest:: wtnetwork

        >>> net = WTNetwork(3)
        >>> net.theta
        <function WTNetwork.split_threshold at 0x...>
        >>> net = WTNetwork(3, theta=WTNetwork.negative_threshold)
        >>> net.theta
        <function WTNetwork.negative_threshold at 0x...>

    This activation function must accept two arguments: the activation stimulus
    and the current state of the node or network. It should handle two types of
    arguments:

        1. stimulus and state are scalar
        2. stimulus and state are vectors (``list`` or :class:`numpy.ndarray`)

    In case 2, the result should `modify` the state in-place and return the vector.

    .. testcode:: wtnetwork

        def theta(stimulus, state):
            if isinstance(stimulus, (list, numpy.ndarray)):
                for i, x in enumerate(stimulus):
                    state[i] = theta(x, state[i])
                return state
            elif stimulus < 0:
                return 0
            else:
                return state
        net = WTNetwork(3, theta=theta)
        print(net.theta)

    .. testoutput:: wtnetwork

        <function theta at 0x...>

    As with all :class:`neet.Network` classes, the names of the nodes and
    network-wide metadata can be provided.

    :param weights: a weights matrix (rows → targets, columns → sources) or a size
    :type weights: int, list, numpy.ndarray
    :param thresholds: activation thresholds for the nodes
    :type thresholds: list, numpy.ndarray
    :param theta: the activation function for all nodes
    :type theta: callable
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    :raises ValueError: if weights is not a integer or a square matrix
    :raises ValueError: if thresholds and weights have inconsistent dimensions
    :raises ValueError: if theta is not callable
    """

    def __init__(self, weights, thresholds=None, theta=None, names=None, metadata=None):
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

        super(WTNetwork, self).__init__(self.thresholds.size, names=names, metadata=metadata)

        if theta is None:
            self.theta = type(self).split_threshold
        elif callable(theta):
            self.theta = theta
        else:
            raise(TypeError("theta must be a function"))

        if self.thresholds.ndim != 1:
            raise(ValueError("thresholds must be a vector"))
        elif shape[0] != self.size:
            msg = "weights and thresholds have different dimensions"
            raise(ValueError(msg))

    def _unsafe_update(self, states, index=None, pin=None, values=None):
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

    @staticmethod
    def read(nodes_path, edges_path, theta=None, metadata=None):
        """
        Read a network from a pair of node/edge files.

        .. doctest:: wtnetwork

            >>> nodes_path = '../neet/boolean/data/s_pombe-nodes.txt'
            >>> edges_path = '../neet/boolean/data/s_pombe-edges.txt'
            >>> net = WTNetwork.read(nodes_path, edges_path)
            >>> net.size
            9
            >>> net.names
            ['SK', 'Cdc2_Cdc13', 'Ste9', 'Rum1', 'Slp1', 'Cdc2_Cdc13_active', 'Wee1_Mik1', 'Cdc25', 'PP']

        :param nodes_path: path to the nodes file
        :type nodes_path: str
        :param edges_path: path to the edges file
        :type edges_path: str
        :param theta: the activation function
        :type theta: callable
        :param metadata: metadata dictionary for the network
        :type metadata: dict
        :return: a :class:`WTNetwork`
        """
        comment = re.compile(r'^\s*#.*$')
        names, thresholds = [], []
        nameindices, index = dict(), 0
        with open(nodes_path, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    name, threshold = line.strip().split()
                    names.append(name)
                    nameindices[name] = index
                    thresholds.append(float(threshold))
                    index += 1

        n = len(names)
        weights = np.zeros((n, n), dtype=np.float)
        with open(edges_path, "r") as f:
            for line in f.readlines():
                if comment.match(line) is None:
                    a, b, w = line.strip().split()
                    weights[nameindices[b], nameindices[a]] = float(w)

        return WTNetwork(weights, thresholds, theta, names=names, metadata=metadata)

    @staticmethod
    def split_threshold(values, states):
        """
        Activates if the stimulus exceeds 0, maintaining state if it is exactly
        0. That is, it is a middle ground between :meth:`negative_threshold`
        and :meth:`positive_threshold`:

        .. math::

            \\theta_s(x,y) = \\begin{cases}
                0 & x < 0 \\\\
                y & x = 0 \\\\
                1 & x > 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y)`` in ``zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples

        .. doctest:: wtnetwork

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
        :return: the updated states
        """
        if isinstance(values, (list, np.ndarray)):
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
        Activate if the stimulus exceeds 0. That is, it "leans negative" if the
        simulus is 0:

        .. math::

            \\theta_n(x) = \\begin{cases}
                0 & x \\leq 0 \\\\
                1 & x > 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y)`` in ``zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples

        .. doctest:: wtnetwork

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
        :return: the updated states
        """
        if isinstance(values, (list, np.ndarray)):
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
        Activate if the stimulus is 0 or greater. That is, it "leans positive"
        if the simulus is 0:

        .. math::

            \\theta_p(x) = \\begin{cases}
                0 & x < 0 \\\\
                1 & x \\geq 0.
            \\end{cases}

        If ``values`` and ``states`` are iterable, then apply the above
        function to each pair ``(x,y)`` in ``zip(values, states)`` and stores
        the result in ``states``.

        If ``values`` and ``states`` are scalar values, then simply apply
        the above threshold function to the pair ``(values, states)`` and
        return the result.

        .. rubric:: Examples

        .. doctest:: wtnetwork

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
            >>> WTNetwork.positive_threshold(0,0)
            1
            >>> WTNetwork.positive_threshold(0,1)
            1
            >>> WTNetwork.positive_threshold(1,0)
            1
            >>> WTNetwork.positive_threshold(-1,0)
            0

        :param values: the threshold-shifted values of each node
        :param states: the pre-updated states of the nodes
        :return: the updated states
        """
        if isinstance(values, (list, np.ndarray)):
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

    def neighbors_in(self, index, *args, **kwargs):
        negative_thresh = type(self).negative_threshold
        positive_thresh = type(self).positive_threshold
        if self.theta is negative_thresh or self.theta is positive_thresh:
            return set(np.flatnonzero(self.weights[index]))
        else:
            # Assume every other theta has self loops. This will be depreciated
            # when we convert all WTNetworks to logicnetworks by default.
            return set(np.flatnonzero(self.weights[index])) | set([index])

    def neighbors_out(self, index, *args, **kwargs):
        negative_thresh = type(self).negative_threshold
        positive_thresh = type(self).positive_threshold
        if self.theta is negative_thresh or self.theta is positive_thresh:
            return set(np.flatnonzero(self.weights[:, index]))

        else:
            # Assume every other theta has self loops. This will be depreciated
            # when we convert all WTNetworks to logicnetworks by default.
            return set(np.flatnonzero(self.weights[:, index])) | set([index])


BooleanNetwork.register(WTNetwork)
