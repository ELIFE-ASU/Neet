"""
.. currentmodule:: neet.boolean

.. testsetup:: sensitivity

    from neet.boolean.examples import c_elegans, s_pombe
    import matplotlib.pyplot as plt
"""
import copy
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


class SensitivityMixin(object):
    """
    SensitivityMixin provides methods for sensitivity analysis. That is,
    methods to quantify the degree to which perturbations of a network's state
    propagate and spread. As part of this, we also provide methods for
    identifying "canalizing edges": edges for which a state of the source node
    uniquely determines the state of the target regardless of other sources.

    .. autosummary::
        :nosignatures:

        sensitivity
        average_sensitivity
        lambdaQ
        difference_matrix
        average_difference_matrix
        is_canalizing
        canalizing_edges
        canalizing_nodes

    The :class:`neet.boolean.BooleanNetwork` class derives from
    SensitivityMixin to provide sensitivity analysis to all of Neet's Boolean
    network models.
    """

    def sensitivity(self, state, transitions=None, timesteps=1):
        """
        Compute the Boolean sensitivity at a given network state.

        The sensitivity of a Boolean function :math:`f` on state vector
        :math:`x` is the number of Hamming neighbors of :math:`x` on which the
        function value is different than on :math:`x`, as defined in
        [Shmulevich2004]_.

        This method calculates the average sensitivity over all :math:`N`
        boolean functions, where :math:`N` is the number of nodes in the
        network.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.sensitivity([0, 0, 0, 0, 0, 1, 1, 0, 0])
            1.0
            >>> s_pombe.sensitivity([0, 1, 1, 0, 1, 0, 0, 1, 0])
            0.4444444444444444
            >>> c_elegans.sensitivity([0, 0, 0, 0, 0, 0, 0, 0])
            1.75
            >>> c_elegans.sensitivity([1, 1, 1, 1, 1, 1, 1, 1])
            1.25

        Optionally, the user can provide a pre-computed array of state
        transitions to improve performance when this function is repeatedly
        called.

        .. doctest:: sensitivity

            >>> trans = list(map(s_pombe.decode, s_pombe.transitions))
            >>> s_pombe.sensitivity([0, 0, 0, 0, 0, 1, 1, 0, 0], transitions=trans)
            1.0
            >>> s_pombe.sensitivity([0, 1, 1, 0, 1, 0, 0, 1, 0], transitions=trans)
            0.4444444444444444

        :param state: a single network state
        :type state: list, numpy.ndarray
        :param transitions: precomputed state transitions (*optional*)
        :type transitions: list, numpy.ndarray, None
        :return: the sensitivity at the provided state

        .. seealso:: :func:`average_sensitivity`
        """
        if not isinstance(timesteps, int):
            raise TypeError('timesteps must be an integer')
        elif timesteps < 0:
            raise ValueError('timesteps must be non-negative')

        if timesteps == 0:
            return 1.0

        encoder = self._unsafe_encode
        update = self._unsafe_update
        distance = self.distance
        neighbors = self.hamming_neighbors(state, c=1)

        nextState = self.update(copy.copy(state))
        for t in range(1, timesteps):
            update(nextState)

        s = 0.
        for neighbor in neighbors:
            for t in range(timesteps):
                if transitions is not None:
                    neighbor = transitions[encoder(neighbor)]
                else:
                    update(neighbor)
            s += distance(neighbor, nextState)

        return s / self.size

    def difference_matrix(self, state, transitions=None):
        """
        Compute the difference matrix at a given state.

        For a network with :math:`N` nodes, with Boolean functions :math:`f_i`,
        the difference matrix is a :math:`N \\times N` matrix

        .. math::

            A_{ij} = f_i(x) \\oplus f_i(x \\oplus e_j)

        where :math:`e_j` is the network state with the :math:`j`-th node in
        the :math:`1` state while all others are :math:`0`. In other words, the
        element :math:`A_{ij}` signifies whether or not flipping the
        :math:`j`-th node's state changes the subsequent state of the
        :math:`i`-th node.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.difference_matrix([0, 0, 0, 0, 0, 0, 0, 0, 0])
            array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 1., 1., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                   [0., 1., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.]])
            >>> c_elegans.difference_matrix([0, 0, 0, 0, 0, 0, 0, 0])
            array([[1., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 1., 1., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 1., 0., 1.],
                   [0., 0., 0., 0., 0., 1., 1., 0.],
                   [1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.]])

        :param state: the starting state
        :type state: list, numpy.ndarray
        :param transitions: precomputed state transitions (*optional*)
        :type transitions: list, numpy.ndarray, None
        :return: the difference matrix

        .. seealso:: :func:`average_difference_matrix`
        """
        N = len(state)
        Q = np.empty((N, N))

        encoder = self._unsafe_encode
        neighbors = self.hamming_neighbors(state)
        nextState = self.update(copy.copy(state))

        for j, neighbor in enumerate(neighbors):
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = self._unsafe_update(neighbor)
            Q[:, j] = [(nextState[i] + newState[i]) % 2 for i in range(N)]

        return Q

    def average_difference_matrix(self, states=None, weights=None, calc_trans=True):
        """

        Compute the difference matrix, averaged over some states.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.average_difference_matrix()
            array([[0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                    0.    ],
                   [0.    , 0.    , 0.25  , 0.25  , 0.25  , 0.    , 0.    , 0.    ,
                    0.    ],
                   [0.25  , 0.25  , 0.25  , 0.    , 0.    , 0.25  , 0.    , 0.    ,
                    0.25  ],
                   [0.25  , 0.25  , 0.    , 0.25  , 0.    , 0.25  , 0.    , 0.    ,
                    0.25  ],
                   [0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.    ,
                    0.    ],
                   [0.    , 0.    , 0.0625, 0.0625, 0.0625, 0.    , 0.0625, 0.0625,
                    0.    ],
                   [0.    , 0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.    ,
                    0.5   ],
                   [0.    , 0.5   , 0.    , 0.    , 0.    , 0.    , 0.    , 0.5   ,
                    0.5   ],
                   [0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    ,
                    0.    ]])
            >>> c_elegans.average_difference_matrix()
            array([[0.25  , 0.25  , 0.    , 0.    , 0.    , 0.25  , 0.25  , 0.25  ],
                   [0.    , 0.    , 0.5   , 0.5   , 0.    , 0.    , 0.    , 0.    ],
                   [0.5   , 0.    , 0.5   , 0.    , 0.5   , 0.    , 0.    , 0.    ],
                   [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                   [0.    , 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.    , 0.3125],
                   [0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.5   , 0.    ],
                   [1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                   [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.5   ]])

        :param states: the states to average over; all states if ``None``
        :type states: list, numpy.ndarray, None
        :param weights: weights for a weighted average over ``states``; uniform
                        weighting if ``None``
        :type weights: list, numpy.ndarray, None
        :param calc_trans: pre-compute all state transitions; ignored if
                           ``states`` or ``weights`` is ``None``
        :type calc_trans: bool
        :return: the difference matrix as a :meth:`numpy.ndarray`.

        .. seealso:: :func:`difference_matrix`
        """
        N = self.size
        Q = np.zeros((N, N))

        if (states is not None) or (weights is not None):
            if calc_trans:
                decoder = self.decode
                trans = list(map(decoder, self.transitions))
            else:
                trans = None

            if states is None:
                states = list(self)
            else:
                states = list(states)

            if weights is None:
                weights = np.ones(len(states))
            else:
                weights = list(weights)

            if np.shape(weights) != (len(states),):
                raise ValueError('Weights must be a 1D array with length same as states')

            norm = np.sum(weights)
            for i, state in enumerate(states):
                Q += weights[i] * self.difference_matrix(state, trans) / norm
        else:
            state0 = np.zeros(N, dtype=int)

            subspace = self.subspace

            for i in range(N):
                nodesInfluencingI = list(self.neighbors_in(i))
                for jindex, j in enumerate(nodesInfluencingI):

                    otherNodes = copy.copy(nodesInfluencingI)
                    otherNodes.pop(jindex)
                    otherNodeStates = list(subspace(otherNodes, state0))
                    for state in otherNodeStates:
                        iState = state[i]
                        state[j] = 0
                        jOffNext = self._unsafe_update(state, index=i)[i]
                        state[i] = iState
                        state[j] = 1
                        jOnNext = self._unsafe_update(state, index=i)[i]
                        Q[i, j] += (jOffNext + jOnNext) % 2
                    Q[i, j] /= float(len(otherNodeStates))

        return Q

    def is_canalizing(self, x, y):
        """
        Determine whether a given network edge is canalizing.

        An edge :math:`(y,x)` is canalyzing if :math:`x`'s value at :math:`t+1`
        is fully determined when :math:`y`'s value has a particular value at
        :math:`t`, regardless of the values of other nodes.

        According to (Stauffer 1987):
        ::

            "A rule [...] is called forcing, or canalizing, if at least one of
            its :math:`K` arguments has the property that the result of the
            function is already fixed if this argument has one particular
            value, regardless of the values for the :math:`K-1` other
            arguments."  Note that this is a definition for whether a node's
            rule is canalizing, whereas this function calculates whether a
            specific edge is canalizing.  Under this definition, if a node has
            any incoming canalizing edges, then its rule is canalizing.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.is_canalizing(1, 2)
            True
            >>> s_pombe.is_canalizing(2, 1)
            False
            >>> c_elegans.is_canalizing(7, 7)
            True
            >>> c_elegans.is_canalizing(1, 3)
            True
            >>> c_elegans.is_canalizing(4, 3)
            False

        :param x: target node's index
        :type x: int
        :param y: source node's index
        :type y: int
        :return: whether or not the edge ``(y,x)`` is canalizing; ``None`` if
                 the edge does not exist

        .. seealso:: :func:`canalizing_edges`, :func:`canalizing_nodes`
        """
        nodesInfluencingI = list(self.neighbors_in(x))

        if (y not in nodesInfluencingI) or (x not in range(self.size)):
            return None
        else:
            jindex = nodesInfluencingI.index(y)

            subspace = self.subspace

            otherNodes = list(copy.copy(nodesInfluencingI))
            otherNodes.pop(jindex)
            otherNodeStates = list(subspace(otherNodes, np.zeros(self.size, dtype=int)))

            jOnForced, jOffForced = True, True
            jOnForcedValue, jOffForcedValue = None, None
            stateindex = 0
            while (jOnForced or jOffForced) and stateindex < len(otherNodeStates):

                state = otherNodeStates[stateindex]

                if jOffForced:
                    jOff = copy.copy(state)
                    jOff[y] = 0
                    jOffNext = self._unsafe_update(jOff, index=x)[x]
                    if jOffForcedValue is None:
                        jOffForcedValue = jOffNext
                    elif jOffForcedValue != jOffNext:
                        jOffForced = False

                if jOnForced:
                    jOn = copy.copy(state)
                    jOn[y] = 1
                    jOnNext = self._unsafe_update(jOn, index=x)[x]
                    if jOnForcedValue is None:
                        jOnForcedValue = jOnNext
                    elif jOnForcedValue != jOnNext:
                        jOnForced = False

                stateindex += 1

            return jOnForced or jOffForced

    def canalizing_edges(self):
        """
        Get the set of all canalizing edges in the network.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.canalizing_edges()
            {(1, 2), (5, 4), (0, 0), (1, 3), (4, 5), (5, 6), (5, 7), (1, 4), (8, 4), (5, 2), (5, 3)}
            >>> c_elegans.canalizing_edges()
            {(1, 2), (3, 2), (1, 3), (7, 6), (6, 0), (7, 7)}

        :return: the set of canalizing edges as in the form ``(target, source)``

        .. seealso:: :func:`is_canalizing`, :func:`canalizing_nodes`
        """
        canalizing_edges = set()
        for x in range(self.size):
            for y in self.neighbors_in(x):
                if self.is_canalizing(x, y):
                    canalizing_edges.add((x, y))
        return canalizing_edges

    def canalizing_nodes(self):
        """
        Get a set of all nodes with at least one incoming canalizing edge.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.canalizing_nodes()
            {0, 1, 4, 5, 8}
            >>> c_elegans.canalizing_nodes()
            {1, 3, 6, 7}

        :return: the set indices of nodes with at least one canalizing input edge

        .. seealso:: :func:`is_canalizing`, :func:`canalizing_edges`
        """
        nodes = [e[0] for e in self.canalizing_edges()]
        return set(np.unique(nodes))

    def lambdaQ(self, **kwargs):
        """
        Compute the sensitivity eigenvalue, :math:`\\lambda_Q`. That is, the
        largest eigenvalue of the sensitivity matrix
        :func:`average_difference_matrix`.

        This is analogous to the eigenvalue calculated in [Pomerance2009]_.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> s_pombe.lambdaQ()
            0.8265021276831896
            >>> c_elegans.lambdaQ()
            1.263099227661824

        :return: the sensitivity eigenvalue (:math:`\\lambda_Q`) of ``net``

        .. seealso:: :func:`average_difference_matrix`
        """
        Q = self.average_difference_matrix(**kwargs)
        return max(abs(linalg.eigvals(Q)))

    def average_sensitivity(self, states=None, weights=None, calc_trans=True, timesteps=1):
        """
        Calculate average Boolean network sensitivity, as defined in
        [Shmulevich2004]_.

        The sensitivity of a Boolean function :math:`f` on state vector :math:`x`
        is the number of Hamming neighbors of :math:`x` on which the function
        value is different than on :math:`x`.

        The average sensitivity is an average taken over initial states.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> c_elegans.average_sensitivity()
            1.265625
            >>> c_elegans.average_sensitivity(states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]])
            ...
            1.5
            >>> c_elegans.average_sensitivity(states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]], weights=[0.9, 0.1])
            ...
            1.7
            >>> c_elegans.average_sensitivity(states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]], weights=[9, 1])
            ...
            1.7

        :param states: The states to average over; all states if ``None``
        :type states: list, numpy.ndarray, None
        :param weights: weights for a weighted average over ``states``; all
                        :math:`1`s if ``None``.
        :type weights: list, numpy.ndarray, None
        :param calc_trans: pre-compute all state transitions; ignored if
                           ``states`` or ``weights`` is ``None``.
        :return: the average sensitivity of ``net``

        .. seealso:: :func:`sensitivity`
        """
        if not isinstance(timesteps, int):
            raise TypeError('timesteps must be an integer')
        elif timesteps < 0:
            raise ValueError('timesteps must be non-negative')

        if timesteps == 0:
            return 1.0
        elif timesteps == 1:
            Q = self.average_difference_matrix(states=states, weights=weights, calc_trans=calc_trans)
            return np.sum(Q) / self.size
        else:
            total = 0.0
            if states is not None:
                for state in states:
                    total += self.sensitivity(state, timesteps=timesteps)
                return total / len(states)
            else:
                total = 0.0
                for state in self:
                    total += self.sensitivity(state, timesteps=timesteps)
                return total / self.volume

    def c_sensitivity(self, state, transitions=None, c=1):
        """
        C-Sensitivity modification of the regular sensitivity function.

        The c-sensitivity of :math:`f(x_1, //ldots, x_n)` at :math:`x` is
        defined as the number of c-Hamming neighbors of :math:`x` on which the
        function value is different from its value on :math:`x`. That is,

        :param state: a single network state
        :type state: list, numpy.ndarray
        :param transitions: precomputed state transitions (*optional*)
        :type transitions: list, numpy.ndarray, None
        :return: the C-sensitivity at the provided state
        """
        if not isinstance(c, int):
            raise TypeError('perturbation size must be an integer')
        elif c < 0:
            raise ValueError('perturbation size must be non-negative')
        elif c > self.size:
            raise ValueError('perturbation size cannot be greater than the number of nodes')
        elif c == 0:
            return 0.0

        encoder = self._unsafe_encode
        update = self._unsafe_update
        distance = self.distance
        nextState = self.update(copy.copy(state))

        s, count = 0.0, 0
        for neighbor in self.hamming_neighbors(state, c):
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = update(neighbor)
            s += distance(newState, nextState)
            count += 1
        return s / count

    def average_c_sensitivity(self, states=None, calc_trans=True, c=1):
        """
        Simple acts as a for-loop which does some precomputation before
        generating all possible states of the network (maintaining topology and
        connections, just changing the initial node values to all possible
        combinations of active nodes) Each generated state's c-sensitivity is
        summed and then divided by the total number of generated states.

        :param states: a set of network states
        :type states: list, numpy.ndarray
        :param calc_trans: pre-compute all state transitions; ignored if
                           ``states`` or ``weights`` is ``None``
        :type calc_trans: bool
        :return: the sensitivity averaged over all possible states of the
                 network
        """
        if not isinstance(c, int):
            raise TypeError('perturbation size must be an integer')
        elif c < 0:
            raise ValueError('perturbation size must be non-negative')
        elif c > self.size:
            raise ValueError('perturbation size cannot be greater than the number of nodes')
        elif c == 0:
            return 0.0

        if calc_trans:
            decoder = self.decode
            trans = list(map(decoder, self.transitions))
        else:
            trans = None

        s = 0
        if states is not None:
            for state in states:
                s += self.c_sensitivity(state, trans, c)
            return s / len(states)
        else:
            for state in self:
                s += self.c_sensitivity(state, trans, c)
            return s / self.volume

    def derrida_plot(self, min_c=0, max_c=None):
        """
        Plot the :math:`c`-sensitivity versus the size of the perturbation :math:`c`.

        :param min_c: minimum perturbation size
        :type min_c: int
        :param max_c: maximum perturbation size
        :type max_c: int
        :return: matplotlib figure and axes
        """
        if max_c is None:
            max_c = self.size + 1

        if not isinstance(min_c, int):
            raise TypeError('minimum perturbation size must be an integer')
        elif min_c < 0:
            raise ValueError('minimum perturbation size must be non-negative')
        elif min_c > self.size:
            raise ValueError('minimum perturbation size cannot be greater than the number of nodes')

        if not isinstance(max_c, int):
            raise TypeError('maximum perturbation size must be an integer')
        elif max_c < 0:
            raise ValueError('maximum perturbation size must be non-negative')
        elif max_c > self.size + 1:
            raise ValueError('maximum perturbation size cannot be greater than the one more than number of nodes')

        if min_c >= max_c:
            raise ValueError('minimum perturbation size must be less than maximum size')

        y_vals = [self.average_c_sensitivity(c=c) for c in range(min_c, max_c)]

        f, ax = plt.subplots()
        ax.set_title('Derrida Plot')
        ax.plot(range(min_c, max_c), y_vals)
        ax.set_xlabel('Pertubation size (c)')
        ax.set_ylabel('Sensitivity')

        return f, ax

    def extended_time_plot(self, min_timesteps=0, max_timesteps=5):
        """
        Plot the sensitivity versus the time since the perturbation.

        :param min_timesteps: minimum number of timesteps
        :type min_timesteps: int
        :param max_timesteps: maximum number of timesteps
        :type max_timesteps: int
        :return: matplotlib figure and axes
        """
        if not isinstance(min_timesteps, int):
            raise TypeError('minimum number of timesteps must be an integer')
        elif min_timesteps < 0:
            raise ValueError('minimum number of timesteps must be non-negative')

        if not isinstance(max_timesteps, int):
            raise TypeError('maximum number of timesteps must be an integer')
        elif max_timesteps < 0:
            raise ValueError('maximum number of timesteps must be non-negative')

        if min_timesteps >= max_timesteps:
            raise ValueError('minimum number of timesteps must be less than maximum number')

        y_vals = [self.average_sensitivity(timesteps=t) for t in range(min_timesteps, max_timesteps)]

        f, ax = plt.subplots()

        ax.set_title('Extended Time Plot')
        ax.plot(range(min_timesteps, max_timesteps), y_vals)
        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('Sensitivity')

        return f, ax
