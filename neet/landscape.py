# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .interfaces import is_network, is_fixed_sized

import copy
import numpy as np
import networkx as nx

class StateSpace(object):
    """
    StateSpace represents the state space of a network model. It may be
    either uniform, i.e. all nodes have the same base, or non-uniform.
    """
    def __init__(self, spec, b=None):
        """
        Initialize the state spec in accordance with the provided ``spec``
        and base ``b``.

        .. rubric:: Examples of Uniform State Spaces:

        ::

            >>> spec = StateSpace(5)
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 5, 2)
            >>> spec = StateSpace(3, b=3)
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 3, 3)
            >>> spec = StateSpace([2,2,2])
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 3, 2)

        .. rubric:: Examples of Non-Uniform State Spaces:

        ::

            >>> spec = StateSpace([2,3,4])
            >>> (spec.is_uniform, spec.bases, spec.ndim)
            (False, [2, 3, 4], 3)

        :param spec: the number of nodes or an array of node bases
        :type spec: int or list
        :param b: the base of the network nodes (ignored if ``spec`` is a list)
        :raises TypeError: if ``spec`` is neither an int nor a list of ints
        :raises TypeError: if ``b`` is neither ``None`` nor an int
        :raises ValueError: if ``b`` is negative or zero
        :raises ValueError: if any element of ``spec`` is negative or zero
        :raises ValueError: if ``spec`` is empty
        """
        if isinstance(spec, int):
            if spec < 1:
                raise(ValueError("ndim cannot be zero or negative"))
            if b is None:
                b = 2
            elif not isinstance(b, int):
                raise(TypeError("base must be an int"))
            elif b < 1:
                raise(ValueError("base must be positive, nonzero"))

            self.is_uniform = True
            self.ndim = spec
            self.base = b
            self.volume = b**spec

        elif isinstance(spec, list):
            if len(spec) == 0:
                raise(ValueError("bases cannot be an empty"))
            else:
                self.is_uniform = True
                self.volume = 1
                base = spec[0]
                if b is not None and base != b:
                    raise(ValueError("b does not match base of spec"))
                for x in spec:
                    if not isinstance(x, int):
                        raise(TypeError("spec must be a list of ints"))
                    elif x < 1:
                        raise(ValueError("spec may only contain positive, nonzero elements"))
                    if self.is_uniform and x != base:
                        self.is_uniform = False
                        if b is not None:
                            raise(ValueError("b does not match base of spec"))
                    self.volume *= x
                self.ndim = len(spec)
                if self.is_uniform:
                    self.base  = base
                else:
                    self.bases = spec[:]
        else:
            raise(TypeError("spec must be an int or a list"))

    def states(self):
        """
        Generate each state of the state space.

        .. rubric:: Examples of Boolean Spaces

        ::

            >>> list(StateSpace(1).states())
            [[0], [1]]
            >>> list(StateSpace(2).states())
            [[0, 0], [1, 0], [0, 1], [1, 1]]
            >>> list(StateSpace(3).states())
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        .. rubric:: Examples of Non-Boolean Spaces

        ::

            >>> list(StateSpace(1,b=3).states())
            [[0], [1], [2]]
            >>> list(StateSpace(2,b=4).states())
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [2, 2], [3, 2], [0, 3], [1, 3], [2, 3], [3, 3]]

        .. rubric:: Examples of Non-Uniform Spaces

        ::

            >>> list(StateSpace([1,2,3]).states())
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]
            >>> list(StateSpace([3,4]).states())
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]]

        :yields: each possible state in the state space
        """
        state = [0] * self.ndim
        yield state[:]
        i = 0
        while i != self.ndim:
            b = self.base if self.is_uniform else self.bases[i]
            if state[i] + 1 < b:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def encode(self, state):
        """
        Encode a state as an integer consistent with the state space.

        .. rubric:: Examples:

        ::

            >>> space = StateSpace(3, b=2)
            >>> states = list(space.states())
            >>> list(map(space.encode, states))
            [0, 1, 2, 3, 4, 5, 6, 7]

        ::

            >>> space = StateSpace([2,3,4])
            >>> states = list(space.states())
            >>> list(map(space.encode, states))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


        :param state: the state to encode
        :type state: list
        :returns: a unique integer encoding of the state
        :raises ValueError: if ``state`` has an incorrect length
        """
        if len(state) != self.ndim:
            raise(ValueError("state has the wrong length"))
        x, q = 0, 1
        if self.is_uniform:
            b = self.base
            for i in range(self.ndim):
                if state[i] < 0 or state[i] >= b:
                    raise(ValueError("invalid node state"))
                x += q * state[i]
                q *= b
        else:
            for i in range(self.ndim):
                b = self.bases[i]
                if state[i] < 0 or state[i] >= b:
                    raise(ValueError("invalid node state"))
                x += q * state[i]
                q *= b
        return x

    def decode(self, x):
        """
        Decode an integer into a state in accordance with the state space.

        .. rubric:: Examples:

        ::

            >>> space = StateSpace(3)
            >>> list(space.states())
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            >>> list(map(space.decode, range(space.volume)))
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        ::

            >>> space = StateSpace([2,3])
            >>> list(space.states())
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
            >>> list(map(space.decode, range(space.volume)))
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]

        :param x: the encoded state
        :type x: int
        :returns: the decoded state as a list
        """
        state = [0] * self.ndim
        if self.is_uniform:
            for i in range(self.ndim):
                state[i] = x % self.base
                x = int(x / self.base)
        else:
            for i in range(self.ndim):
                state[i] = x % self.bases[i]
                x = int(x / self.bases[i])
        return state


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
    """
    if is_network(net):
        g = transition_graph(net)
    elif isinstance(net,nx.DiGraph):
        g = net
    else:
        raise TypeError("net must be a network or a networkx DiGraph")

    return nx.weakly_connected_component_subgraphs(g)





