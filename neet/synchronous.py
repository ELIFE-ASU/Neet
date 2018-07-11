# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import copy
import networkx as nx
import numpy as np
import pyinform as pi
from .statespace import StateSpace
from .interfaces import is_network, is_fixed_sized

def trajectory(net, state, timesteps=1, encode=False):
    """
    Generate the trajectory of length ``timesteps+1`` through the state-space,
    as determined by the network rule, beginning at ``state``.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> gen = trajectory(ECA(30), [0, 1, 0], timesteps=3)
        >>> gen
        <generator object trajectory at 0x000002B692ED8BF8>
        >>> list(gen)
        [[0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]]
        >>> list(trajectory(ECA(30), [0, 1, 0], timesteps=3, encode=True))
        [2, 7, 0, 0]
        >>> gen = trajectory(s_pombe, [0, 0, 0, 0, 1, 0, 0, 0, 0], timesteps=3,
        ... encode=True)
        >>> list(gen)
        [16, 256, 78, 128]

    :param net: the network
    :param state: the network state
    :param timesteps: the number of steps in the trajectory
    :param encode: encode the states as integers
    :returns: the trajectory as a list
    :raises TypeError: if net is not a network
    :raises ValueError: if ``timesteps < 1``
    """
    if not is_network(net):
        raise TypeError("net is not a network")
    if timesteps < 1:
        raise ValueError("number of steps must be positive, non-zero")

    traj = []
    state = copy.copy(state)
    if encode:
        if is_fixed_sized(net):
            state_space = net.state_space()
        else:
            state_space = net.state_space(len(state))

        traj.append(state_space._unsafe_encode(state))

        net.update(state)
        traj.append(state_space._unsafe_encode(state))

        for _ in range(1,timesteps):
            net._unsafe_update(state)
            traj.append(state_space._unsafe_encode(state))
    else:
        traj.append(copy.copy(state))

        net.update(state)
        traj.append(copy.copy(state))

        for _ in range(1, timesteps):
            net._unsafe_update(state)
            traj.append(copy.copy(state))
    return traj

def transitions(net, size=None, encode=False):
    """
    Generate the one-step state transitions for a network over its state space.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> gen = transitions(ECA(30), size=3)
        >>> gen
        <generator object transitions at 0x000002B691328BA0>
        >>> list(gen)
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1],
        [0, 1, 0], [0, 0, 0]]
        >>> list(transitions(ECA(30), size=3, encode=True))
        [0, 7, 7, 1, 7, 4, 2, 0]
        >>> gen = transitions(s_pombe, encode=True)
        >>> len(list(gen))
        512

    :param net: the network
    :param size: the size of the network (``None`` if fixed sized)
    :param encode: encode the states as integers
    :returns: the one-state transitions as an array
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    if not is_network(net):
        raise TypeError("net is not a network")

    if is_fixed_sized(net):
        if size is not None:
            raise ValueError("size must be None for fixed sized networks")
        state_space = net.state_space()
    else:
        if size is None:
            raise ValueError("size must not be None for variable sized networks")
        state_space = net.state_space(size)

    trans = []
    for state in state_space:
        net._unsafe_update(state)
        if encode:
            trans.append(state_space._unsafe_encode(state))
        else:
            trans.append(state)

    return trans

def transition_graph(net, size=None):
    """
    Construct the state transition graph for the network.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> g = transition_graph(s_pombe)
        >>> g.number_of_nodes(), g.number_of_edges()
        (512, 512)
        >>> g = transition_graph(ECA(30), size=6)
        >>> g.number_of_nodes(), g.number_of_edges()
        (64, 64)

    :param net: the network (if already a networkx.DiGraph, does nothing and returns it)
    :param size: the size of the network (``None`` if fixed sized)
    :param encode: encode the states as integers
    :returns: a ``networkx.DiGraph`` of the network's transition graph
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    if is_network(net):
        edge_list = enumerate(transitions(net, size=size, encode=True))
        return nx.DiGraph(list(edge_list))
    elif isinstance(net, nx.DiGraph):
        if size is not None:
            raise ValueError("size must be None for transition graphs")
        return net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")

def attractors(net, size=None):
    """
    Find the attractor states of a network. A generator of the attractors is
    returned with each attractor represented as a ``list`` of "encoded" states.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> list(attractors(s_pombe))
        [[76], [4], [8], [12], [144, 110, 384], [68], [72], [132], [136],
        [140], [196], [200], [204]]
        >>> list(attractors(ECA(30), size=5))
        [[0], [14, 25, 7, 28, 19]]

    :param net: the network or the transition graph
    :param size: the size of the network (``None`` if fixed sized)
    :returns: a list of attractor cycles
    :raises TypeError: if ``net`` is not a network or a ``networkx.DiGraph``
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is a transition graph and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    if isinstance(net, nx.DiGraph):
        return list(nx.simple_cycles(net))
    elif not is_network(net):
        raise TypeError("net must be a network or a networkx DiGraph")
    elif is_fixed_sized(net) and size is not None:
        raise ValueError("fixed sized networks require size is None")
    elif not is_fixed_sized(net) and size is None:
        raise ValueError("variable sized networks require a size")
    else:
        cycles = []
        # Get the state transitions
        # (array of next state indexed by current state)
        trans = list(transitions(net, size=size, encode=True))
        # Create an array to store whether a given state has visited
        visited = np.zeros(len(trans), dtype=np.bool)
        # Create an array to store which attractor basin each state is in
        basins = np.zeros(len(trans), dtype=np.int)
        # Create a counter to keep track of how many basins have been visited
        basin_number = 1

        # Start at state 0
        initial_state = 0
        # While the initial state is a state of the system
        while initial_state < len(trans):
            # Create a stack to store the state so far visited
            state_stack = []
            # Create a array to store the states in the attractor cycle
            cycle = []
            # Create a flag to signify whether the current state is part of the cycle
            in_cycle = False
            # Set the current state to the initial state
            state = initial_state
            # Store the next state and terminus variables to the next state
            terminus = next_state = trans[state]
            # Set the visited flag of the current state
            visited[state] = True
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

            # If the next state hasn't been assigned a basin yet
            if basins[next_state] == 0:
                # Set the current basin to the basin number
                basin = basin_number
                # Add the current state to the attractor cycle
                cycle.append(state)
                # We're still in the cycle until the current state is equal to the terminus
                in_cycle = (terminus != state)
            else:
                # Set the current basin to the basin of next_state
                basin = basins[next_state]

            # Set the basin of the current state
            basins[state] = basin

            # While we still have states on the stack
            while len(state_stack) != 0:
                # Pop the current state off of the top of the stack
                state = state_stack.pop()
                # Set the basin of the current state
                basins[state] = basin
                # If we're still in the cycle
                if in_cycle:
                    # Add the current state to the attractor cycle
                    cycle.append(state)
                    # We're still in the cycle until the current state is equal to the terminus
                    in_cycle = (terminus != state)

            # Find the next unvisited initial state
            while initial_state < len(visited) and visited[initial_state]:
                initial_state += 1

            # Yield the cycle if we found one
            if len(cycle) != 0:
                cycles.append(cycle)
    return cycles

def basins(net, size=None):
    """
    Find the attractor basins of a network. A generator of the attractor basins
    is returned with each basin represented as a ``networkx.DiGraph`` whose
    nodes are the "encoded" network states.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> b = basins(s_pombe)
        >>> [len(basin) for basin in b]
        [378, 2, 2, 2, 104, 6, 6, 2, 2, 2, 2, 2, 2]
        >>> b = basins(ECA(30), size=5)
        >>> [len(basin) for basin in b]
        [2, 30]

    :param net: the network or landscape transition_graph
    :param size: the size of the network (``None`` if fixed sized)
    :returns: generator of basin subgraphs
    :raises TypeError: if ``net`` is not a network or a ``networkx.DiGraph``
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is a transition graph and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    graph = transition_graph(net, size=size)
    return nx.weakly_connected_component_subgraphs(graph)

def basin_entropy(net, size=None, base=2):
    """
    Calculate the basin entropy [Krawitz2007]_.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> basin_entropy(s_pombe)
        1.2218888338849747
        >>> basin_entropy(s_pombe, base=10)
        0.367825190366261
        >>> basin_entropy(ECA(30), size=5)
        0.3372900666170139

    :param net: the network or landscape transition_graph
    :param size: the size of the network (``None`` if fixed sized)
    :param base: base of logarithm used to calculate entropy (2 for bits)
    :returns: value of basin entropy
    :raises TypeError: if ``net`` is not a network or a ``networkx.DiGraph``
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is a transition graph and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    sizes = [ len(basin) for basin in basins(net, size=size) ]
    d = pi.Dist(sizes)
    return pi.shannon.entropy(d, b=base)

def timeseries(net, timesteps, size=None):
    """
    Return the timeseries for the network. The result will be a :math:`3D` array
    with shape :math:`N \\times V \\times t` where :math:`N` is the number of
    nodes in the network, :math:`V` is the volume of the state space (total
    number of network states), and :math:`t` is ``timesteps + 1``.

    ::

        >>> net = WTNetwork([[1,-1],[1,0]])
        >>> timeseries(net, 5)
        array([[[ 0.,  0.,  0.,  0.,  0.,  0.],
                [ 1.,  1.,  1.,  1.,  1.,  1.],
                [ 0.,  0.,  0.,  0.,  0.,  0.],
                [ 1.,  1.,  1.,  1.,  1.,  1.]],

            [[ 0.,  0.,  0.,  0.,  0.,  0.],
                [ 0.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.]]])

    :param net: the network
    :param timesteps: the number of timesteps in the timeseries
    :param size: the size of the network (``None`` if fixed sized)
    :return: a numpy array
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    :raises ValueError: if ``timesteps < 1``
    """
    if not is_network(net):
        raise TypeError("net must be a NEET network")
    if not is_fixed_sized(net) and size is None:
        raise ValueError("network is not fixed sized; must provide a size")
    elif is_fixed_sized(net) and size is not None:
        raise ValueError("cannot provide a size with a fixed sized network")
    if timesteps < 1:
        raise ValueError("time series must have at least one timestep")

    if size is None:
        state_space = net.state_space()
    else:
        state_space = net.state_space(size)

    shape = (state_space.ndim, state_space.volume, timesteps+1)
    series = np.empty(shape, dtype=np.int)

    trans = list(transitions(net, size=size, encode=False))
    encoded_trans = [state_space._unsafe_encode(state) for state in trans]

    for (index, init) in enumerate(state_space):
        k = index
        series[:, index, 0] = init[:]
        for time in range(1, timesteps + 1):
            series[:, index, time] = trans[k][:]
            k = encoded_trans[k]

    return series

class Landscape(StateSpace):
    """
    The ``Landscape`` class represents the structure and topology of the
    "landscape" of state transitions. That is, it is the state space
    together with information about state transitions and the topology of
    the state transition graph.
    """
    def __init__(self, net, size=None):
        """
        Construct the landscape for a network.

        .. rubric:: Example:

        ::

            >>> from neet.automata import ECA
            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            <neet.synchronous.Landscape object at 0x101c74810>
            >>> Landscape(ECA(30), size=5)
            <neet.synchronous.Landscape object at 0x10415b6d0>

        :param net: the network
        :param size: the size of the network (``None`` if fixed sized)
        :raises TypeError: if ``net`` is not a network
        :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
        :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
        """

        if not is_network(net):
            raise TypeError("net is not a network")
        elif is_fixed_sized(net):
            if size is not None:
                raise ValueError("size must be None for fixed sized networks")
            state_space = net.state_space()
        else:
            if size is None:
                raise ValueError("size must not be None for variable sized networks")
            state_space = net.state_space(size)

        if state_space.is_uniform:
            super(Landscape, self).__init__(state_space.ndim, state_space.base)
        else:
            super(Landscape, self).__init__(state_space.bases)

        self.__net = net

        self.__expounded = False
        self.__graph = None

        self.__setup()

    @property
    def network(self):
        """
        The landscape's dynamical network

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.network
            array([array([76]), array([4]), array([8]), array([12]),
                   array([144, 110, 384]), array([68]), array([72]), array([132]),
                   array([136]), array([140]), array([196]), array([200]), array([204])], dtype=object)
        """
        return self.__net

    @property
    def size(self):
        """
        The number of nodes in the landscape's dynamical network

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.attractors
            array([array([76]), array([4]), array([8]), array([12]),
                   array([144, 110, 384]), array([68]), array([72]), array([132]),
                   array([136]), array([140]), array([196]), array([200]), array([204])], dtype=object)
        """
        if not self.__expounded:
            self.__expound()
        return self.__attractors

    @property
    def basins(self):
        """
        The array of basin numbers, indexed by states.

        This array associates each state with its basin number.

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.heights
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 1, 1, 2, 8, 1,
                   1, 2, 8, 1, 1, 2, 8, 1, 1, 2, 2, 2, 2, 9, 9, 1, 1, 9, 9, 1, 1, 9, 9,
                   ...
                   2, 3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.recurrence_times
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 3, 3, 2, 8, 3,
                   3, 2, 8, 3, 3, 2, 8, 3, 3, 4, 4, 4, 4, 9, 9, 3, 3, 9, 9, 3, 3, 9, 9,
                   ...
                   4, 3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
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

        transitions = np.empty(self.volume, dtype=np.int)
        for i, state in enumerate(self):
            transitions[i] = encode(update(state))

        self.__transitions = transitions

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
            - :py:method:`.attractors`
            - :py:method:`.basins`
            - :py:method:`.basin_sizes`
            - :py:method:`.in_degrees`
            - :py:method:`.heights`
            - :py:method:`.attractor_lengths`
            - :py:method:`.recurrence_times`
        """
        # Get the state transitions
        trans = self.__transitions
        # Create an array to store whether a given state has visited
        visited = np.zeros(self.volume, dtype=np.bool)
        # Create an array to store which attractor basin each state is in
        basins = np.full(self.volume, -1, dtype=np.int)
        # Create an array to store the in-degree of each state
        in_degrees = np.zeros(self.volume, dtype=np.int)
        # Create an array to store the height of each state
        heights = np.zeros(self.volume, dtype=np.int)
        # Create an array to store the recurrence time of each state
        recurrence_times = np.zeros(self.volume, dtype=np.int)
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
            # Create a flag to signify whether the current state is part of the cycle
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
                # We're still in the cycle until the current state is equal to the terminus
                in_cycle = (terminus != state)
            else:
                # Set the current basin to the basin of next_state
                basin = basins[next_state]
                # Set the state's height to one greater than the next state's
                heights[state] = heights[next_state] + 1
                # Set the state's recurrence time to one greater than the next state's
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
                    # We're still in the cycle until the current state is equal to the terminus
                    in_cycle = (terminus != state)
                    # Set the cycle state's recurrence times
                    if not in_cycle:
                        for cycle_state in cycle:
                            recurrence_times[cycle_state] = attractor_lengths[basin] - 1
                else:
                    # Set the state's height to one create than the next state's
                    heights[state] = heights[next_state] + 1
                    # Set the state's recurrence time to one greater than the next state's
                    recurrence_times[state] = recurrence_times[next_state] + 1

            # Find the next unvisited initial state
            while initial_state < len(visited) and visited[initial_state]:
                initial_state += 1

            # If the cycle isn't empty, append it to the attractors list
            if len(cycle) != 0:
                attractors.append(np.asarray(cycle, dtype=np.int))

        self.__basins = basins
        self.__basin_sizes = np.asarray(basin_sizes)
        self.__attractors = np.asarray(attractors)
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

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.trajectory([1,0,0,1,0,1,1,0,1])
            [[1, 0, 0, 1, 0, 1, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 1],
             [0, 1, 1, 1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1, 0, 1, 0],
             [0, 1, 0, 0, 1, 1, 0, 1, 0],
             [0, 0, 0, 0, 1, 0, 0, 1, 1],
             [0, 0, 1, 1, 0, 0, 1, 0, 1],
             [0, 0, 1, 1, 0, 0, 1, 0, 0]]

            >>> landscape.trajectory([1,0,0,1,0,1,1,0,1], encode=True)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> landscape.trajectory(361)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> landscape.trajectory(361, encode=False)
            [[1, 0, 0, 1, 0, 1, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 1],
             [0, 1, 1, 1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1, 0, 1, 0],
             [0, 1, 0, 0, 1, 1, 0, 1, 0],
             [0, 0, 0, 0, 1, 0, 0, 1, 1],
             [0, 0, 1, 1, 0, 0, 1, 0, 1],
             [0, 0, 1, 1, 0, 0, 1, 0, 0]]

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
                path[i] = trans[path[i-1]]
        else:
            path = [init]
            state = trans[init]
            while state not in path:
                path.append(state)
                state = trans[state]

        if not encode:
            decode = self.decode
            path = [ decode(state) for state in path ]

        return path

    def timeseries(self, timesteps):
        """
        Compute the full time series of the landscape.

        This method computes a 3-dimensional array elements are the
        states of each node in the network. The dimensions of the array
        are indexed by, in order, the node, the initial state and the
        time step.

        .. rubric:: Example:

        ::

            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.timeseries(5)
            array([[[0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],

                   [[0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0],
                    ...,
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],

                   ...

                   [[0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0]],

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
        encode = self._unsafe_encode
        decoded_trans = [ decode(state) for state in trans ]
        encoded_trans = [ encode(state) for state in decoded_trans ]

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
        number of basins so that the result is :math:`0 \leq h \leq 1`.
        If there is fewer than :math:`2` basins, then the base is taken
        to be :math:`2` so that the result is never `NaN`. The base can
        be forcibly overridden with the ``base`` keyword argument.

        .. rubric:: Example:

        ::

            >>> from math import e
            >>> from neet.boolean.examples import s_pombe
            >>> from neet.synchronous import Landscape
            >>> landscape = Landscape(s_pombe)
            >>> landscape.basin_entropy()
            1.2218888338849747
            >>> landscape.basin_entropy(base=2)
            1.2218888338849747
            >>> landscape.basin_entropy(base=10)
            0.367825190366261
            >>> landscape.basin_entropy(base=e)
            0.8469488001650496

        :param base: the base of the logarithm
        :type base: a number or ``None``
        :return: the basin entropy of the landscape of type ``float``
        """
        if not self.__expounded:
            self.__expound()
        dist = pi.Dist(self.__basin_sizes)
        return pi.shannon.entropy(dist, b=base)
