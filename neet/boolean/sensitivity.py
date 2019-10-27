"""
.. currentmodule:: neet.sensitivity

.. testsetup:: sensitivity

    from neet.boolean.examples import c_elegans, s_pombe
    from neet.sensitivity import *

Sensitivity
===========

The :mod:`neet.sensitivity` module provides a collection of functions for
computing measures of sensitivity of networks, i.e. the degree to which
perturbations of the network state propogate and spread. This module also
provides a collection of functions for identifying "canalizing edges": edges
for which a state of the source node uniquely determines the state of the
target regardless of other sources.

API Documentation
-----------------
"""
import copy
import numpy as np
import numpy.linalg as linalg


class SensitivityMixin:
    def sensitivity(self, state, transitions=None):
        """
        Calculate Boolean network sensitivity, as defined in [Shmulevich2004]_

        The sensitivity of a Boolean function :math:`f` on state vector :math:`x`
        is the number of Hamming neighbors of :math:`x` on which the function
        value is different than on :math:`x`.

        This calculates the average sensitivity over all :math:`N` boolean
        functions, where :math:`N` is the size of self.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> sensitivity(s_pombe, [0, 0, 0, 0, 0, 1, 1, 0, 0])
            1.0
            >>> sensitivity(s_pombe, [0, 1, 1, 0, 1, 0, 0, 1, 0])
            0.4444444444444444
            >>> sensitivity(c_elegans, [0, 0, 0, 0, 0, 0, 0, 0])
            1.75
            >>> sensitivity(c_elegans, [1, 1, 1, 1, 1, 1, 1, 1])
            1.25

        :param state: a single network state, represented as a list of node states
        :param transitions: a list of precomputed state transitions (*optional*)
        :type transitions: list or None
        """
        encoder = self._unsafe_encode
        distance = self.distance
        neighbors = self.hamming_neighbors(state)

        nextState = self.update(state)

        # count sum of differences found in neighbors of the original
        s = 0.
        for neighbor in neighbors:
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = self._unsafe_update(neighbor)
            s += distance(newState, nextState)

        return s / self.size

    def difference_matrix(self, state, transitions=None):
        """
        Returns matrix answering the question: Starting at the given state, does
        flipping the state of node ``j`` change the state of node ``i``?

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> difference_matrix(s_pombe, [0, 0, 0, 0, 0, 0, 0, 0, 0])
            array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 1., 1., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 1.],
                   [0., 1., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.]])
            >>> difference_matrix(c_elegans, [0, 0, 0, 0, 0, 0, 0, 0])
            array([[1., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 1., 1., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 1., 0., 1.],
                   [0., 0., 0., 0., 0., 1., 1., 0.],
                   [1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1.]])

        :param state: the starting state
        :param transitions: a precomputed list of state transitions (*optional*)
        :type transitions: list or None
        """
        # set up empty matrix
        N = len(state)
        Q = np.empty((N, N))

        # list Hamming neighbors (in order!)
        encoder = self._unsafe_encode
        neighbors = self.hamming_neighbors(state)

        nextState = self.update(state)

        # count differences found in neighbors of the original
        for j, neighbor in enumerate(neighbors):
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = self._unsafe_update(neighbor)
            Q[:, j] = [(nextState[i] + newState[i]) % 2 for i in range(N)]

        return Q

    def average_difference_matrix(self, states=None, weights=None, calc_trans=True):
        """
        Averaged over states, what is the probability
        that node i's state is changed by a single bit flip of node j?

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> average_difference_matrix(s_pombe)
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
            >>> average_difference_matrix(c_elegans)
            array([[0.25  , 0.25  , 0.    , 0.    , 0.    , 0.25  , 0.25  , 0.25  ],
                   [0.    , 0.    , 0.5   , 0.5   , 0.    , 0.    , 0.    , 0.    ],
                   [0.5   , 0.    , 0.5   , 0.    , 0.5   , 0.    , 0.    , 0.    ],
                   [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                   [0.    , 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.    , 0.3125],
                   [0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.5   , 0.    ],
                   [1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                   [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.5   ]])

        :param states: If None, average over all possible states. Otherwise,
                       providing a list of states will calculate the average over
                       only those states.
        :type states: list or None
        :param calc_trans: Optionally pre-calculate all transitions. Only used
                            when states or weights argument is not None.
        :type calc_trans: bool
        :return: boolean ``numpy`` array
        """
        N = self.size
        Q = np.zeros((N, N))

        if (states is not None) or (weights is not None):
            # explicitly calculate difference matrix for each state

            # optionally pre-calculate transitions
            if calc_trans:
                decoder = self.decode
                trans = list(map(decoder, self.transitions))
            else:
                trans = None

            # currently changes state generators to lists.
            # is there a way to avoid this?
            if states is None:
                states = list(self)
            else:
                states = list(states)

            if weights is None:
                weights = np.ones(len(states))
            else:
                weights = list(weights)

            if np.shape(weights) != (len(states),):
                msg = "Weights must be a 1D array with length same as states"
                raise(ValueError(msg))

            norm = np.sum(weights)
            for i, state in enumerate(states):
                Q += weights[i] * self.difference_matrix(state, trans) / norm

        else:  # make use of sparse connectivity to be more efficient
            state0 = np.zeros(N, dtype=int)

            subspace = self.subspace

            for i in range(N):
                nodesInfluencingI = list(self.neighbors_in(i))
                for jindex, j in enumerate(nodesInfluencingI):

                    # for each state of other nodes, does j matter?
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
                        # are the results different?
                        Q[i, j] += (jOffNext + jOnNext) % 2
                    Q[i, j] /= float(len(otherNodeStates))

        return Q

    def is_canalizing(self, node_i, neighbor_j):
        """
        Determine whether a given network edge is canalizing: if ``node_i``'s
        value at :math:`t+1` is fully determined when ``neighbor_j``'s value has
        a particular value at :math:`t`, regardless of the values of other nodes,
        then there is a canalizing edge from ``neighbor_j`` to node_i.

        According to (Stauffer 1987), "A rule ... is called forcing, or
        canalizing, if at least one of its :math:`K` arguments has the property
        that the result of the function is already fixed if this argument has
        one particular value, regardless of the values for the :math:`K-1` other
        arguments."  Note that this is a definition for whether a node's rule is
        canalizing, whereas this function calculates whether a specific edge is
        canalizing.  Under this definition, if a node has any incoming canalizing
        edges, then its rule is canalizing.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> is_canalizing(s_pombe, 1, 2)
            True
            >>> is_canalizing(s_pombe, 2, 1)
            False
            >>> is_canalizing(c_elegans, 7, 7)
            True
            >>> is_canalizing(c_elegans, 1, 3)
            True
            >>> is_canalizing(c_elegans, 4, 3)
            False

        :param node_i: target node index
        :param neighbor_j: source node index
        :return: ``True`` if the edge ``(neighbor_j, node_i)`` is canalizing, or
                 ``None`` if the edge does not exist

        .. seealso:: :func:`canalizing_edges`
        .. seealso:: :func:`canalizing_nodes`
        """
        nodesInfluencingI = list(self.neighbors_in(node_i))

        if (neighbor_j not in nodesInfluencingI) or (node_i not in range(self.size)):
            # can't be canalizing if j has no influence on i
            return None  # or False?
        else:
            jindex = nodesInfluencingI.index(neighbor_j)

            subspace = self.subspace

            # for every state of other nodes, does j determine i?
            otherNodes = list(copy.copy(nodesInfluencingI))
            otherNodes.pop(jindex)
            otherNodeStates = list(subspace(otherNodes, np.zeros(self.size, dtype=int)))

            jOnForced, jOffForced = True, True
            jOnForcedValue, jOffForcedValue = None, None
            stateindex = 0
            while (jOnForced or jOffForced) and stateindex < len(otherNodeStates):

                state = otherNodeStates[stateindex]

                # first hold j off
                if jOffForced:
                    jOff = copy.copy(state)
                    jOff[neighbor_j] = 0
                    jOffNext = self._unsafe_update(jOff, index=node_i)[node_i]
                    if jOffForcedValue is None:
                        jOffForcedValue = jOffNext
                    elif jOffForcedValue != jOffNext:
                        # then holding j off does not force i
                        jOffForced = False

                # now hold j on
                if jOnForced:
                    jOn = copy.copy(state)
                    jOn[neighbor_j] = 1
                    jOnNext = self._unsafe_update(jOn, index=node_i)[node_i]
                    if jOnForcedValue is None:
                        jOnForcedValue = jOnNext
                    elif jOnForcedValue != jOnNext:
                        # then holding j on does not force i
                        jOnForced = False

                stateindex += 1

            # if we have checked all states, then the edge must be forcing
            # print "jOnForced,jOffForced",jOnForced,jOffForced
            return jOnForced or jOffForced

    def canalizing_edges(self):
        """
        Return a set of tuples corresponding to the edges in the network that
        are canalizing. Each tuple consists of two node indices, corresponding
        to an edge from the second node to the first node (so that the second node
        controls the first node in a canalizing manner).

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> canalizing_edges(s_pombe)
            {(1, 2), (5, 4), (0, 0), (1, 3), (4, 5), (5, 6), (5, 7), (1, 4), (8, 4), (5, 2), (5, 3)}
            >>> canalizing_edges(c_elegans)
            {(1, 2), (3, 2), (1, 3), (7, 6), (6, 0), (7, 7)}

        :return: the set of canalizing edges as in the form ``(target, source)``

        .. seealso:: :func:`is_canalizing`
        .. seealso:: :func:`canalizing_nodes`
        """
        canalizingList = []
        for indexi in range(self.size):
            for neighborj in self.neighbors_in(indexi):
                if self.is_canalizing(indexi, neighborj):
                    canalizingList.append((indexi, neighborj))
        return set(canalizingList)

    def canalizing_nodes(self):
        """
        Find the nodes of the network which have at least one incoming canalizing
        edge.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> canalizing_nodes(s_pombe)
            {0, 1, 4, 5, 8}
            >>> canalizing_nodes(c_elegans)
            {1, 3, 6, 7}

        :return: the set indices of nodes with at least one canalizing input edge

        .. seealso:: :func:`is_canalizing`
        .. seealso:: :func:`canalizing_edges`
        """
        nodes = [e[0] for e in self.canalizing_edges()]
        return set(np.unique(nodes))

    def lambdaQ(self, **kwargs):
        """
        Calculate sensitivity eigenvalue, the largest eigenvalue of the
        sensitivity matrix :func:`average_difference_matrix`.

        This is analogous to the eigenvalue calculated in [Pomerance2009]_.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> lambdaQ(s_pombe)
            0.8265021276831896
            >>> lambdaQ(c_elegans)
            1.263099227661824

        :return: the sensitivity eigenvalue (:math:`\\lambda_Q`) of ``net``
        """
        Q = self.average_difference_matrix(**kwargs)
        return max(abs(linalg.eigvals(Q)))

    def average_sensitivity(self, states=None, weights=None, calc_trans=True):
        """
        Calculate average Boolean network sensitivity, as defined in
        [Shmulevich2004]_.

        The sensitivity of a Boolean function :math:`f` on state vector :math:`x`
        is the number of Hamming neighbors of :math:`x` on which the function
        value is different than on :math:`x`.

        The average sensitivity is an average taken over initial states.

        .. rubric:: Examples

        .. doctest:: sensitivity

            >>> average_sensitivity(c_elegans)
            1.265625
            >>> average_sensitivity(c_elegans, states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]])
            ...
            1.5
            >>> average_sensitivity(c_elegans, states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]],weights=[0.9, 0.1])
            ...
            1.7
            >>> average_sensitivity(c_elegans, states=[[0, 0, 0, 0, 0, 0, 0, 0],
            ... [1, 1, 1, 1, 1, 1, 1, 1]], weights=[9, 1])
            ...
            1.7

        :param states: Optional list or generator of states. If None, all states
                       are used.
        :param weights: Optional list or generator of weights for each state.
                        If None, each state is equally weighted. If states and
                        weights are both None, an algorithm is used to efficiently
                        make use of sparse connectivity.
        :return: the average sensitivity of ``net``
        """

        Q = self.average_difference_matrix(states=states, weights=weights,
                                           calc_trans=calc_trans)

        return np.sum(Q) / self.size

    def C_sensitivity(self, state, c=1, transitions=None):
        """C-Sensitivity modification of the regular sensitivity function. I deleted the 
        doctest code because it was cluttering my screen"""

        """The c-sensitivity of f(x1, . . ., xn) at x is defined as the number of 
        c-Hamming neighbors of x on which the function value is different from its value on x. That is,"""

        encoder = self._unsafe_encode
        distance = self.distance
        neighbors = self.hamming_neighbors(state)

        nextState = self.update(state)

        # count sum of differences found in neighbors of the original
        s = 0.
        for neighbor in neighbors:
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = self._unsafe_update(neighbor)
            s += distance(newState, nextState)

        return s / self.size