"""
.. currentmodule:: neet.boolean

.. testsetup:: sensitivity

    from neet.boolean.examples import c_elegans, s_pombe
"""
import copy
import numpy as np
import numpy.linalg as linalg
import math
import itertools as itt


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

    def sensitivity(self, state, transitions=None):
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
            # can't be canalizing if j has no influence on i
            return None  # or False?
        else:
            jindex = nodesInfluencingI.index(y)

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
                    jOff[y] = 0
                    jOffNext = self._unsafe_update(jOff, index=x)[x]
                    if jOffForcedValue is None:
                        jOffForcedValue = jOffNext
                    elif jOffForcedValue != jOffNext:
                        # then holding j off does not force i
                        jOffForced = False

                # now hold j on
                if jOnForced:
                    jOn = copy.copy(state)
                    jOn[y] = 1
                    jOnNext = self._unsafe_update(jOn, index=x)[x]
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

        Q = self.average_difference_matrix(states=states, weights=weights,
                                           calc_trans=calc_trans)

        return np.sum(Q) / self.size

    def c_sensitivity(self, state, transitions=None, c=1):

        assert !(c < 0),"the value of c needs to be greater than or equal to zero"
        assert !(c > self.size),"the value of c needs to be between 0 and the size of the network"
        assert (isinstance(c, int)),"c needs to be an integer"


        """C-Sensitivity modification of the regular sensitivity function. I deleted the 
        doctest code because it was cluttering my screen

        The c-sensitivity of :math:`f(x_1, //ldots, x_n)` at :math:`x` is defined as the number of 
        c-Hamming neighbors of :math:`x` on which the function value is different from its value on :math:`x`. That is,

        :param state: a single network state
        :type state: list, numpy.ndarray
        :param transitions: precomputed state transitions (*optional*)
        :type transitions: list, numpy.ndarray, None
        :return: the C-sensitivity at the provided state

        """

        encoder = self._unsafe_encode
        distance = self.distance
        state_copy = copy.copy(state)
        nextState = self.update(state)

        """
        Returns an iterator for each vector I which is a strict subset of {1,...,n} and where |I| = c
        note: if c = 0, this will return an empty tuple
        """

        I_comb_iter = itt.combinations(range(self.size), c)

        """ 
        Generator function which returns a new hamming neighbor
        Each hamming neighbor is simply the product of self.state XOR I
        """
        def c_hamming_neighbors(self, state, c):
            try:
                nxt = next(I_comb_iter)
                XORed = copy.copy(state_copy)
                for i in nxt:
                    XORed[i] ^= 1
                return XORed
            except StopIteration:
                return None


        """ 
        #OK, so I messed with the function and it's only ~kinda~ a generator function now...
        It's automatically advanced in the for loop, which
        acts as a "try: next(neighbors); catch StopIteration:". This behavior is built 
        into Python and is idiomatic.
        """ 

        s = 0.
        neighbors_copy = []#uses quite a bit of memory and should be removed from code once unit testing is completed
        copy_counter = 0
    
        neighbor = c_hamming_neighbors(self,state,c)
        while neighbor is not None:
            neighbors_copy.append(copy.copy(neighbor))
            if transitions is not None:
                newState = transitions[encoder(neighbor)]
            else:
                newState = self._unsafe_update(neighbor)

            # the paper which describes c-sensitivity uses an indicator function
            # instead of a distance function. Distance will be used here instead of the indicator.
            
            if distance(newState, nextState) > 0:
                if c == 0:
                    # The distance between 0-hamming neighbors should
                    # be 0, since a 0-hamming neighbor is just the
                    # original state/vector
                    print("This shouldn't print. 0-hamming neighbors should not diverge.")
                    print("c: ", c)
                    print("1. neighbor:  ", neighbors_copy[copy_counter])
                    print("2. state:     ", state_copy,"\n")
                    print("1. newState:  ", newState)
                    print("2. nextState: ", nextState,"\n\n")
                s += distance(newState, nextState)
            copy_counter += 1
            neighbor = c_hamming_neighbors(self,state,c)
        return s / copy_counter


    def average_c_sensitivity(self, states=None, calc_trans=True, c=1):

        """
        Simple acts as a for-loop which does some precomputation before generating 
        all possible states of the network (maintaining topology and connections, 
        just changing the initial node values to all possible combinations of active
        nodes)
        Each generated state's c-sensitivity is summed and then divided by the total
        number of generated states.
        """


        """

        :param states: a set of network states
        :type states: list, numpy.ndarray
        :param transitions: precomputed state transitions (*optional*)
        :type transitions: list, numpy.ndarray, None
        :param calc_trans: pre-compute all state transitions; ignored if
                           ``states`` or ``weights`` is ``None``
        :type calc_trans: bool
        :return: the sensitivity averaged over all possible 
                            states of the network


        """


        s = 0

        # optionally pre-calculate transitions
        if calc_trans:
            decoder = self.decode
            trans = list(map(decoder, self.transitions))
        else:
            trans = None


        if states is not None:
            # Iterate through all provided states and sum 
            # the distances between them and their c-hamming-neighbors
            # after a single time-step (one synchronous transition)
            for state in states:
                s += self.c_sensitivity(state, trans, c)

            # get the average of s by diving the sum by the
            # number of states considered
            s = s / len(states)


        else:
            # Generate all possible states 
            # and sum the distances between them and each of their
            # c-hamming-neighbors in the next time step (after one 
            # synchronous transition)
            for state in self:
                s += self.c_sensitivity(state, trans, c)


            # Deprecated implementation I'm keeping around while we implement unit testing. Will be removed after
            # equivalence/correctness of the above (2 line) implementation is established
            """for n in range(self.size):
                state_gen = itt.combinations(range(self.size),n)
                for state in state_gen:
                    state_array = [0 for x in range(self.size)]
                    for index in state:
                        state_array[index] = 1
                    s += self.c_sensitivity(state_array, trans, c)"""

            # Average s by diving the sum by the 
            # total number of possible states (2^n)
            # s = s / np.power(2, self.size)
            s = s / self.volume
            """ 
            s is now the average C-Sensitivity of f and must lie in the interval [0, (n choose c)] 
            where n is the size of the network.
            """

            upper_bound = math.factorial(self.size) / (math.factorial(c) * math.factorial(n - c))
            if s > upper_bound or s < 0:
                raise RuntimeError('This value of S should not be possible and the code is therefore wrong')

        return s
