"""
.. currentmodule:: neet.synchronous

.. testsetup:: synchronous

    from math import e
    from neet.automata import ECA
    from neet.boolean import WTNetwork
    from neet.boolean.examples import s_pombe
    from neet.synchronous import *

Synchronous Network Analysis
============================

The disadvantage of these functions is that they cannot share any
information. If you wish to compute a number of them, then many values will be
repeatedly computed. For this reason, we created the :class:`Landscape` class
to simultaneously compute and several of these properties. This provides much
better performance overall.

API Documentation
-----------------
"""
import networkx as nx
import numpy as np
import pyinform as pi
import copy
from .statespace import StateSpace
from .network import Network


class Landscape(StateSpace):
    """
    The ``Landscape`` class represents the structure and topology of the
    "landscape" of state transitions. That is, it is the state space
    together with information about state transitions and the topology of
    the state transition graph.
    """

    def __init__(self, net, index=None, pin=None, values=None,
                 dynamic_pin=None):
        """
        Construct the landscape for a network.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> Landscape(s_pombe)
            <neet.synchronous.Landscape object at 0x...>
            >>> Landscape(ECA(30, 5))
            <neet.synchronous.Landscape object at 0x...>

        :param net: the network
        :param index: the index to update (or None)
        :param pin: the indices to pin during update (or None); see also
            dynamic_pin
        :param values: a dictionary of index-value pairs to set after update
        :param dynamic_pin: a list of lists of pinned values over time;
            shape = (# timesteps)x(# pinned nodes)
        :raises TypeError: if ``net`` is not a network
        :raises ValueError: if ``dynamic_pin`` has incorrect shape
        """

        if not isinstance(net, Network):
            raise TypeError("net is not a network")
        else:
            state_space = net.state_space()
        
        if dynamic_pin is not None:
            if pin is None:
                raise ValueError("pin must be specified to use dynamic_pin")
            elif len(dynamic_pin[0]) != len(pin):
                raise ValueError("each element of dynamic_pin must have the same length as pin")

        super(Landscape, self).__init__(state_space.shape)

        self.__net = net
        self.__index = index
        self.__pin = pin
        self.__values = values
        self.__dynamic_pin = dynamic_pin

        self.__expounded = False
        self.__graph = None

        self.__setup()

    @property
    def network(self):
        """
        The landscape's dynamical network

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.network
            <neet.boolean.wtnetwork.WTNetwork object at 0x...>
        """
        return self.__net

    @property
    def size(self):
        """
        The number of nodes in the landscape's dynamical network

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.size
            9
        """
        return self.ndim

    @property
    def transitions(self):
        """
        The state transitions array

        The transitions array is an array, indexed by states, whose
        values are the subsequent state of the indexing state.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.transitions
            array([  2,   2, 130, 130,   4,   0, 128, 128,   8,   0, 128, 128,  12,
                     0, 128, 128, 256, 256, 384, 384, 260, 256, 384, 384, 264, 256,
                   ...
                   208, 208, 336, 336, 464, 464, 340, 336, 464, 464, 344, 336, 464,
                   464, 348, 336, 464, 464])
        """
        return self.__transitions

    @property
    def attractors(self):
        """
        The array of attractor cycles.

        Each element of the array is an array of states in an attractor
        cycle.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.attractors
            array([array([76]), array([4]), array([8]), array([12]),
                   array([144, 110, 384]), array([68]), array([72]), array([132]),
                   array([136]), array([140]), array([196]), array([200]),
                   array([204])], dtype=object)
        """
        if not self.__expounded:
            self.__expound()
        return self.__attractors

    @property
    def basins(self):
        """
        The array of basin numbers, indexed by states.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.basins
            array([ 0,  0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  0,  3,  0,  0,  0,  0,
                    0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  4,  4,
                    ...
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0])
        """
        if not self.__expounded:
            self.__expound()
        return self.__basins

    @property
    def basin_sizes(self):
        """
        The basin sizes as an array indexed by the basin number.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.basin_sizes
            array([378,   2,   2,   2, 104,   6,   6,   2,   2,   2,   2,   2,   2])
        """
        if not self.__expounded:
            self.__expound()
        return self.__basin_sizes

    @property
    def in_degrees(self):
        """
        The in-degree of each state as an array.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.in_degrees
            array([ 6,  0,  4,  0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  0, 12,
                    0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    ...
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0])
        """
        if not self.__expounded:
            self.__expound()
        return self.__in_degrees

    @property
    def heights(self):
        """
        The height of each state as an array.

        The *height* of a state is the number of time steps from that
        state to a state in it's attractor cycle.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.heights
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 1, 1, 2, 8,
                   1, 1, 2, 8, 1, 1, 2, 8, 1, 1, 2, 2, 2, 2, 9, 9, 1, 1, 9, 9, 1, 1,
                   ...
                   3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3])
        """
        if not self.__expounded:
            self.__expound()
        return self.__heights

    @property
    def attractor_lengths(self):
        """
        The length of each attractor cycle as an array, indexed by the
        attractor.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.attractor_lengths
            array([1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1])
        """
        if not self.__expounded:
            self.__expound()
        return self.__attractor_lengths

    @property
    def recurrence_times(self):
        """
        The recurrence time of each state as an array.

        The *recurrence time* is the number of time steps from that
        state until a state is repeated.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.recurrence_times
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 3, 3, 2, 8,
                   3, 3, 2, 8, 3, 3, 2, 8, 3, 3, 4, 4, 4, 4, 9, 9, 3, 3, 9, 9, 3, 3,
                   ...
                   3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3])
        """
        if not self.__expounded:
            self.__expound()
        return self.__recurrence_times

    @property
    def graph(self):
        """
        The state transitions graph of the landscape as a
        ``networkx.Digraph``.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.graph
            <networkx.classes.digraph.DiGraph object at 0x106504810>
        """
        if self.__graph is None:
            self.__graph = nx.DiGraph(list(enumerate(self.__transitions)))
        return self.__graph

    def __setup(self):
        """
        This function performs all of the initilization-time setup of
        the ``Landscape`` object. At present this is limited to
        computing the state transitions array, but subsequent versions
        may expand the work that ``__setup`` does.
        """
        update = self.__net._unsafe_update
        encode = self._unsafe_encode

        if self.__dynamic_pin is None:
            transitions = np.empty(self.volume, dtype=np.int)
            for i, state in enumerate(self):
                transitions[i] = encode(update(state,
                                               index=self.__index,
                                               pin=self.__pin,
                                               values=self.__values))
        else: # update multiple times using dynamic pin for inputs
            
            # we will construct a special limited state space that does not
            # iterate over fixed nodes
            limited_shape = self.shape[:]
            for fixed_node_index in self.__pin:
                limited_shape[fixed_node_index] = 1
            self.limited_state_space = StateSpace(limited_shape)
            
            # we will keep track of both the "transitions" (starting points
            # to endpoints after ell timesteps, where ell is the length of
            # the dynamic pin) and "dynamic_paths" (all steps along the way).
            # note that transitions are encoded using the limited state space,
            # while dynamic paths are encoded using the full state space.
            transitions = np.empty(self.limited_state_space.volume, dtype=np.object)
            dynamic_paths = np.empty((self.limited_state_space.volume,
                                      len(self.__dynamic_pin)),
                                     dtype=np.object)
            for i,state in enumerate(self.limited_state_space):
                current_state = copy.copy(state)
                ell = len(self.__dynamic_pin)
                # set first pinned values
                for k,pinned_value in enumerate(self.__dynamic_pin[-1]):
                    current_state[self.__pin[k]] = pinned_value
                for j in range(ell):
                    # update with those nodes pinned
                    current_state = update(current_state,
                                           index=self.__index,
                                           pin=self.__pin,
                                           values=self.__values)
                    # set to next pinned values
                    for k,pinned_value in enumerate(self.__dynamic_pin[j%ell]):
                        current_state[self.__pin[k]] = pinned_value
                    # record result
                    dynamic_paths[i,j] = encode(current_state)
                # reset pinned values to zero to encode correctly using limited state space
                for pinned_node in self.__pin:
                        current_state[pinned_node] = 0
                transitions[i] = self.limited_state_space._unsafe_encode(current_state)

        self.__transitions = transitions
        if self.__dynamic_pin is None:
            self.__dynamic_paths = transitions.reshape(len(transitions),1)
        else:
            self.__dynamic_paths = dynamic_paths
        self.dynamic_paths = self.__dynamic_paths # THERE'S PROBABLY A BETTER WAY

    def __expound(self):
        """
        This function performs the bulk of the calculations that the
        ``Landscape`` is concerned with. Most of the properties in this
        class are computed by this function whenever *any one* of them
        is requested and the results are cached. The advantage of this
        is that it saves computation time; why traverse the state space
        for every property call when you can do it all at once. The
        downside is that the cached results may use a good bit more
        memory. This is a trade-off that we are willing to make for now.

        The properties that are computed by this function include:
            - :method:`attractors`
            - :method:`basins`
            - :method:`basin_sizes`
            - :method:`in_degrees`
            - :method:`heights`
            - :method:`attractor_lengths`
            - :method:`recurrence_times`
        """
        
        if self.__dynamic_pin is None:
            volume = self.volume
        else:
            # With a dynamic pin, the volume of state space to be
            # explored is limited
            volume = self.limited_state_space.volume
        
        # Get the state transitions
        trans = self.__transitions
        # Create an array to store whether a given state has visited
        visited = np.zeros(volume, dtype=np.bool)
        # Create an array to store which attractor basin each state is in
        basins = np.full(volume, -1, dtype=np.int)
        # Create an array to store the in-degree of each state
        in_degrees = np.zeros(volume, dtype=np.int)
        # Create an array to store the height of each state
        heights = np.zeros(volume, dtype=np.int)
        # Create an array to store the recurrence time of each state
        recurrence_times = np.zeros(volume, dtype=np.int)
        # Create a counter to keep track of how many basins have been visited
        basin_number = 0
        # Create a list of basin sizes
        basin_sizes = []
        # Create a list of attractor cycles
        attractors = []
        # Create a list of attractor lengths
        attractor_lengths = []

        # Start at state 0
        initial_state = 0
        # While the initial state is a state of the system
        while initial_state < len(trans):
            # Create a stack to store the state so far visited
            state_stack = []
            # Create a array to store the states in the attractor cycle
            cycle = []
            # Create a flag to signify whether the current state is part of
            # the cycle
            in_cycle = False
            # Set the current state to the initial state
            state = initial_state
            # Store the next state and terminus variables to the next state
            terminus = next_state = trans[state]
            # Set the visited flag of the current state
            visited[state] = True
            # Increment in-degree
            in_degrees[next_state] += 1
            # While the next state hasn't been visited
            while not visited[next_state]:
                # Push the current state onto the stack
                state_stack.append(state)
                # Set the current state to the next state
                state = next_state
                # Update the terminus and next_state variables
                terminus = next_state = trans[state]
                # Update the visited flag for the current state
                visited[state] = True
                # Increment in-degree
                in_degrees[next_state] += 1

            # If the next state hasn't been assigned a basin yet
            if basins[next_state] == -1:
                # Set the current basin to the basin number
                basin = basin_number
                # Increment the basin number
                basin_number += 1
                # Add a new basin size
                basin_sizes.append(0)
                # Add a new attractor length
                attractor_lengths.append(1)
                # Add the current state to the attractor cycle
                cycle.append(state)
                # Set the current state's recurrence time
                recurrence_times[state] = 0
                # We're still in the cycle until the current state is equal to
                # the terminus
                in_cycle = (terminus != state)
            else:
                # Set the current basin to the basin of next_state
                basin = basins[next_state]
                # Set the state's height to one greater than the next state's
                heights[state] = heights[next_state] + 1
                # Set the state's recurrence time to one greater than the next
                # state's
                recurrence_times[state] = recurrence_times[next_state] + 1

            # Set the basin of the current state
            basins[state] = basin
            # Increment the basin size
            basin_sizes[basin] += 1

            # While we still have states on the stack
            while len(state_stack) != 0:
                # Save the current state as the next state
                next_state = state
                # Pop the current state off of the top of the stack
                state = state_stack.pop()
                # Set the basin of the current state
                basins[state] = basin
                # Increment the basin_size
                basin_sizes[basin] += 1
                # If we're still in the cycle
                if in_cycle:
                    # Add the current state to the attractor cycle
                    cycle.append(state)
                    # Increment the current attractor length
                    attractor_lengths[basin] += 1
                    # We're still in the cycle until the current state is
                    # equal to the terminus
                    in_cycle = (terminus != state)
                    # Set the cycle state's recurrence times
                    if not in_cycle:
                        for cycle_state in cycle:
                            rec_time = attractor_lengths[basin] - 1
                            recurrence_times[cycle_state] = rec_time
                else:
                    # Set the state's height to one create than the next
                    # state's
                    heights[state] = heights[next_state] + 1
                    # Set the state's recurrence time to one greater than the
                    # next state's
                    recurrence_times[state] = recurrence_times[next_state] + 1

            # Find the next unvisited initial state
            while initial_state < len(visited) and visited[initial_state]:
                initial_state += 1

            # If the cycle isn't empty, append it to the attractors list
            if len(cycle) != 0:
                attractors.append(np.asarray(cycle, dtype=np.int))

        # BCD 12.19.2018 I think the attractors are drawkcab...
        attractors = [ a[::-1] for a in attractors ]

        self.__basins = basins
        self.__basin_sizes = np.asarray(basin_sizes)
        if self.__dynamic_pin is None:
            self.__attractors = np.asarray(attractors)
        else:
            # each attractor state needs to be expanded to length ell paths
            attractor_paths = [ np.concatenate([ self.dynamic_paths[state] \
                                                 for state in a]) \
                                for a in attractors ]
            self.__attractors = np.asarray(attractor_paths)
        self.__attractor_lengths = np.asarray(attractor_lengths)
        self.__in_degrees = in_degrees
        self.__heights = heights
        self.__recurrence_times = np.asarray(recurrence_times)
        self.__expounded = True

    def trajectory(self, init, timesteps=None, encode=None):
        """
        Compute the trajectory of a state.

        This method computes a trajectory from ``init`` to the last
        before the trajectory begins to repeat. If ``timesteps`` is
        provided, then the trajectory will have a length of ``timesteps
        + 1`` regardless of repeated states. The ``encode`` argument
        forces the states in the trajectory to be either encoded or not.
        When ``encode is None``, whether or not the states of the
        trajectory are encoded is determined by whether or not the
        initial state (``init``) is provided in encoded form.

        Note that when ``timesteps is None``, the length of the
        resulting trajectory should be one greater than the recurrence
        time of the state.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.trajectory([1,0,0,1,0,1,1,0,1])
            [[1, 0, 0, 1, 0, 1, 1, 0, 1], ... [0, 0, 1, 1, 0, 0, 1, 0, 0]]

            >>> landscape.trajectory([1,0,0,1,0,1,1,0,1], encode=True)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> landscape.trajectory(361)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> landscape.trajectory(361, encode=False)
            [[1, 0, 0, 1, 0, 1, 1, 0, 1], ... [0, 0, 1, 1, 0, 0, 1, 0, 0]]

            >>> landscape.trajectory(361, timesteps=5)
            [361, 80, 320, 78, 128, 162]

            >>> landscape.trajectory(361, timesteps=10)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76, 76]

        :param init: the initial state
        :type init: ``int`` or an iterable
        :param timesteps: the number of time steps to include in the trajectory
        :type timesteps: ``int`` or ``None``
        :param encode: whether to encode the states in the trajectory
        :type encode: ``bool`` or ``None``
        :return: a list whose elements are subsequent states of the trajectory

        :raises ValueError: if ``init`` an empty array
        :raises ValueError: if ``timesteps`` is less than :math:`1`
        """
        decoded = isinstance(init, list) or isinstance(init, np.ndarray)

        if decoded:
            if init == []:
                raise ValueError("initial state cannot be empty")
            elif encode is None:
                encode = False
            init = self.encode(init)
        elif encode is None:
            encode = True

        trans = self.__transitions
        if timesteps is not None:
            if timesteps < 1:
                raise ValueError("number of steps must be positive, non-zero")

            path = [init] * (timesteps + 1)
            for i in range(1, len(path)):
                path[i] = trans[path[i - 1]]
        else:
            path = [init]
            state = trans[init]
            while state not in path:
                path.append(state)
                state = trans[state]

        if not encode:
            decode = self.decode
            path = [decode(state) for state in path]

        return path

    def timeseries(self, timesteps):
        """
        Compute the full time series of the landscape.

        This method computes a 3-dimensional array elements are the
        states of each node in the network. The dimensions of the array
        are indexed by, in order, the node, the initial state and the
        time step.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.timeseries(5)
            array([[[0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0],
                    ...,
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],
            <BLANKLINE>
                   ...
            <BLANKLINE>
                   [[0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1],
                    ...,
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0]]])

        :param timesteps: the number of timesteps to evolve the system
        :type timesteps: ``int``
        :return: a 3-D array of node states

        :raises ValueError: if ``timesteps`` is less than :math:`1`
        """
        if timesteps < 1:
            raise ValueError("number of steps must be positive, non-zero")

        trans = self.__transitions
        decode = self.decode
        decoded_trans = [decode(state) for state in trans]

        shape = (self.ndim, self.volume, timesteps + 1)
        series = np.empty(shape, dtype=np.int)

        for index, init in enumerate(self):
            k = index
            series[:, index, 0] = init[:]
            for time in range(1, timesteps + 1):
                series[:, index, time] = decoded_trans[k][:]
                k = trans[k]

        return series

    def basin_entropy(self, base=2.0):
        """
        Compute the basin entropy of the landscape [Krawitz2007]_.

        This method computes the Shannon entropy of the distribution of
        basin sizes. The base of the logarithm is chosen to be the
        number of basins so that the result is :math:`0 \\leq h \\leq 1`.
        If there is fewer than :math:`2` basins, then the base is taken
        to be :math:`2` so that the result is never `NaN`. The base can
        be forcibly overridden with the ``base`` keyword argument.

        .. rubric:: Examples

        .. doctest:: synchronous

            >>> landscape = Landscape(s_pombe)
            >>> landscape.basin_entropy()
            1.221888833884975
            >>> landscape.basin_entropy(base=2)
            1.221888833884975
            >>> landscape.basin_entropy(base=10)
            0.36782519036626105
            >>> landscape.basin_entropy(base=e)
            0.8469488001650497

        :param base: the base of the logarithm
        :type base: a number or ``None``
        :return: the basin entropy of the landscape of type ``float``
        """
        if not self.__expounded:
            self.__expound()
        dist = pi.Dist(self.__basin_sizes)
        return pi.shannon.entropy(dist, b=base)
