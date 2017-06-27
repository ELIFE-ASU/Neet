# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import copy
import networkx as nx
import numpy as np
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
    :yields: the next state in the trajectory
    :raises TypeError: if net is not a network
    :raises ValueError: if ``timesteps < 1``
    """
    if not is_network(net):
        raise TypeError("net is not a network")
    if timesteps < 1:
        raise ValueError("number of steps must be positive, non-zero")

    state = copy.copy(state)
    if encode:
        if is_fixed_sized(net):
            space = net.state_space()
        else:
            space = net.state_space(len(state))

        yield space.encode(state)
        for _ in range(timesteps):
            net.update(state)
            yield space.encode(state)
    else:
        yield copy.copy(state)
        for _ in range(timesteps):
            net.update(state)
            yield copy.copy(state)

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
    :yields: the one-state transitions
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    if not is_network(net):
        raise TypeError("net is not a network")

    if is_fixed_sized(net):
        if size is not None:
            raise ValueError("size must be None for fixed sized networks")
        space = net.state_space()
    else:
        if size is None:
            raise ValueError("size must not be None for variable sized networks")
        space = net.state_space(size)

    for state in space.states():
        net.update(state)
        if encode:
            yield space.encode(state)
        else:
            yield state

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

    :param net: the network
    :param size: the size of the network (``None`` if fixed sized)
    :param encode: encode the states as integers
    :returns: a ``networkx.DiGraph`` of the network's transition graph
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    edge_list = enumerate(transitions(net, size=size, encode=True))
    return nx.DiGraph(list(edge_list))

def attractors(net, size=None):
    """
    Find the attractor states of a network. A generator of the attractors is
    returned with each attractor represented as a ``list`` of "encoded" states.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> from neet.boolean.examples import s_pombe
        >>> list(attractors(s_pombe))
        [[204], [200], [196], [140], [136], [132], [72], [68], [384, 110, 144],
        [12], [8], [4], [76]]
        >>> list(attractors(ECA(30), size=5))
        [[7, 25, 14, 19, 28], [0]]

    :param net: the network or the transition graph
    :param size: the size of the network (``None`` if fixed sized)
    :returns: a generator of attractors
    :raises TypeError: if ``net`` is not a network or a ``networkx.DiGraph``
    :raises ValueError: if ``net`` is fixed sized and ``size`` is not ``None``
    :raises ValueError: if ``net`` is a transition graph and ``size`` is not ``None``
    :raises ValueError: if ``net`` is not fixed sized and ``size`` is ``None``
    """
    if is_network(net):
        graph = transition_graph(net, size=size)
    elif isinstance(net, nx.DiGraph):
        if size is not None:
            raise ValueError("size must be None for transition graphs")
        graph = net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")

    return nx.simple_cycles(graph)

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
    if is_network(net):
        graph = transition_graph(net, size=size)
    elif isinstance(net, nx.DiGraph):
        graph = net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")

    return nx.weakly_connected_component_subgraphs(graph)

def timeseries(net, timesteps, size=None):
    """
    Return the timeseries for the network. The result will be a :math:`3D` array
    with shape :math:`N times V times t` where :math:`N` is the number of nodes
    in the network, :math:`V` is the volume of the state space (total number of
    network states), and :math:`t` is ``timesteps + 1``.

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
    for (index, init) in enumerate(state_space.states()):
        traj = trajectory(net, init, timesteps=timesteps, encode=False)
        for (time, state) in enumerate(traj):
            series[:, index, time] = state
    return series
