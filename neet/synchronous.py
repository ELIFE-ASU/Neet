# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_network, is_fixed_sized
from .statespace import StateSpace

import copy
import numpy as np
import networkx as nx

def trajectory(net, state, n=1, encode=False):
    """
    Generate the trajectory of length ``n+1`` through state-space, as determined
    by the network rule, beginning at ``state``.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> gen = trajectory(ECA(30), [0,1,0], n=3)
        >>> gen
        <generator object trajectory at 0x000001DF6E02B888>
        >>> list(gen)
        [[0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]]
        >>> list(trajectory(ECA(30), [0,1,0], n=3, encode=True))
        [2,7,0,0]

    :param net: the network
    :param state: the network state
    :param n: the number of steps in the trajectory
    :param encode: encode the states as integers
    :yields: the ``n+1`` states in the trajectory
    :raises TypeError: ``not is_network(net)``
    :raises ValueError: if ``n < 1``
    """
    if not is_network(net):
        raise(TypeError("net is not a network"))
    if n < 1:
        raise(ValueError("number of steps must be positive, non-zero"))

    state = copy.copy(state)
    if encode:
        if is_fixed_sized(net):
            space = net.state_space()
        else:
            space = net.state_space(len(state))

        yield space.encode(state)
        for i in range(n):
            net.update(state)
            yield space.encode(state)
    else:
        yield copy.copy(state)
        for i in range(n):
            net.update(state)
            yield copy.copy(state)


def transitions(net, n=None, encode=True):
    """
    Generate the one-step state transitions for a network over its state space.

    .. rubric:: Example:

    ::

        >>> from neet.automata import ECA
        >>> gen = transitions(ECA(30), n=3)
        >>> gen
        <generator object transitions at 0x000001DF6E02B938>
        >>> list(gen)
        [0, 7, 7, 1, 7, 4, 2, 0]
        >>> list(transitions(ECA(30), n=3, encode=False))
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0,
        1], [0, 1, 0], [0, 0, 0]]

    :param net: the network
    :param n: the number of nodes in the network
    :type n: ``None`` or ``int``
    :param encode: encode the states as integers
    :type encode: boolean
    :yields: the one-state transitions
    :raises TypeError: if ``net`` is not a network
    :raises TypeError: if ``space`` is not a :class:`neet.StateSpace`
    """
    if not is_network(net):
        raise(TypeError("net is not a network"))

    if is_fixed_sized(net):
        if n is not None:
            raise(TypeError("n must be None for fixed sized networks"))
        space = net.state_space()
    else:
        space = net.state_space(n)

    if not isinstance(space, StateSpace):
        raise(TypeError("network's state space is not an instance of StateSpace"))

    for state in space.states():
        net.update(state)
        if encode:
            yield space.encode(state)
        else:
            yield state


def transition_graph(net, n=None):
    """
    Return a networkx graph representing net's transition network.
    
    .. rubric:: Example:
    
    ::
    
        >>> from neet.boolean.examples import s_pombe
        >>> g = landscape.transition_graph(s_pombe)
        >>> g.number_of_edges()
        512
    """

    if not is_network(net):
        raise(TypeError("net is not a network"))

    edgeList = enumerate( transitions(net,n) )
    
    return nx.DiGraph(edgeList)

def attractors(net):
    """
    Return a generator that lists net's attractors.  Each attractor 
    is represented as a list of 'encoded' states.
    
    .. rubric:: Example:
    
    ::
    
        >>> from neet.boolean.examples import s_pombe
        >>> print(list(attractors(s_pombe)))
        [[204], [200], [196], [140], [136], [132], [72], [68], 
        [384, 110, 144], [12], [8], [4], [76]]
        
    :param net: the network or landscape transition_graph
    :type net: neet network or networkx DiGraph
    :returns: generator of attractors
    :raises TypeError: if ``net`` is not a network or DiGraph
    """
    if is_network(net):
        g = transition_graph(net)
    elif isinstance(net,nx.DiGraph):
        g = net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")
    
    return nx.simple_cycles(g)

def basins(net):
    """
    Return a generator that lists net's basins.  Each basin
    is a networkx graph.
    
    .. rubric:: Example:
    
    ::
    
    :param net: the network or landscape transition_graph
    :type net: neet network or networkx DiGraph
    :returns: generator of basin subgraphs
    :raises TypeError: if ``net`` is not a network or DiGraph
    """
    if is_network(net):
        g = transition_graph(net)
    elif isinstance(net,nx.DiGraph):
        g = net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")

    return nx.weakly_connected_component_subgraphs(g)

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
    :type net: neet network
    :param timesteps: the number of timesteps in the timeseries
    :type timesteps: int
    :param size: the size of the network (if it is variable-sized)
    :type size: int
    :return: a numpy array
    :raises TypeError: if ``net`` is not a network
    :raises ValueError: if ``net`` is not fixed-sized and ``size`` is ``None``
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
        traj = trajectory(net, init, n=timesteps, encode=False)
        for (time, state) in enumerate(traj):
            series[:, index, time] = state
    return series