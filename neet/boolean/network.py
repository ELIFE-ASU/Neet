"""
.. currentmodule:: neet.boolean

.. testsetup:: boolean_network

    from neet.boolean.examples import s_pombe
"""
from neet import UniformNetwork
from neet.python import long
from .sensitivity import SensitivityMixin
import copy


class BooleanNetwork(SensitivityMixin, UniformNetwork):
    """
    The BooleanNetwork class is a base class for all of Neet's Boolean
    networks. The BooleanNetwork class inherits from both
    :class:`neet.UniformNetwork` and :class:`neet.boolean.SensitivityMixin`,
    and specializes the inherited :class:`neet.StateSpace` methods to exploit
    the Boolean structure.

    .. inheritance-diagram:: neet.boolean.BooleanNetwork
        :parts: 1

    In addition to all of its inherited methods, BooleanNetwork also exposes the following methods:

    .. autosummary::
        :nosignatures:

        subspace
        distance
        hamming_neighbors

    BooleanNetwork is an *abstract* class, meaning it cannot be instantiated.
    Initialization of a BooleaNetwork requires, at a minimum, the number of
    nodes in the network. As with all classes that derive from
    :class:`neet.Network`, the user may optionally provide a list of names for
    the nodes of the network and a metadata dictionary for the network as a
    whole (e.g. citation information).

    :param size: number of nodes in the network
    :type size: int
    :param names: an iterable object of the names of the nodes in the network
    :type names: seq
    :param metadata: metadata dictionary for the network
    :type metadata: dict
    """

    def __init__(self, size, names=None, metadata=None):
        super(BooleanNetwork, self).__init__(size, 2, names, metadata)

    def __iter__(self):
        size = self.size
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            if state[i] == 0:
                state[i] = 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, state):
        try:
            if len(state) != self.size:
                return False

            for x in state:
                if x != 0 and x != 1:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)
        for x in state:
            encoded += place * long(x)
            place <<= 1
        return encoded

    def decode(self, encoded):
        size = self.size
        state = [0] * size
        for i in range(size):
            state[i] = encoded & 1
            encoded >>= 1
        return state

    def subspace(self, indices, state=None):
        """
        Generate all states in a given subspace. This method varies each node
        specified by the ``indicies`` array independently. The optional
        ``state`` parameter specifies the state of the non-varying states of
        the network. If ``state`` is not provided, all nodes not in
        ``indicies`` will have state ``0``.

        .. rubric:: Examples

        .. doctest:: boolean_network

            >>> s_pombe.subspace([0])
            <generator object BooleanNetwork.subspace at 0x...>
            >>> list(s_pombe.subspace([0]))
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]
            >>> list(s_pombe.subspace([0, 3]))
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0]]

        .. doctest:: boolean_network

            >>> s_pombe.subspace([0], state=[0, 1, 0, 1, 0, 1, 0, 1, 0])
            <generator object BooleanNetwork.subspace at 0x...>
            >>> list(s_pombe.subspace([0], state=[0, 1, 0, 1, 0, 1, 0, 1, 0]))
            [[0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 1, 0]]
            >>> list(s_pombe.subspace([0, 3], state=[0, 1, 0, 1, 0, 1, 0, 1, 0]))
            [[0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0]]

        :param indicies: the indicies to vary in the subspace
        :type indicies: list, numpy.ndarray, iterable
        :param state: a state which specifes the state of the non-varying nodes
        :type state: list, numpy.ndarray
        :yield: the states of the subspace
        """
        size = self.size

        if state is not None and state not in self:
            raise ValueError('provided state is not in the state space')
        elif state is None:
            state = [0] * size

        indices = list(set(indices))
        indices.sort()
        nindices = len(indices)

        if nindices == 0:
            yield copy.copy(state)
        elif indices[0] < 0 or indices[-1] >= size:
            raise IndexError('index out of range')
        elif nindices == size:
            for state in self:
                yield state
        else:

            initial = copy.copy(state)

            yield copy.copy(state)
            i = 0
            while i != nindices:
                if state[indices[i]] == initial[indices[i]]:
                    state[indices[i]] ^= 1
                    for j in range(i):
                        state[indices[j]] = initial[indices[j]]
                    i = 0
                    yield copy.copy(state)
                else:
                    i += 1

    def hamming_neighbors(self, state):
        """
        Get all states that one unit of Hamming distance from a given state.

        .. rubric:: Examples

        .. doctest:: boolean_network

            >>> s_pombe.hamming_neighbors([0, 0, 0, 0, 0, 0, 0, 0, 0])
            [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]
            >>> s_pombe.hamming_neighbors([0, 1, 1, 0, 1, 0, 1, 0, 0])
            [[1, 1, 1, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1]]

        :param state: the state whose neighbors are desired
        :type state: list, numpy.ndarray
        :return: a list of neighbors of the given state
        :raises ValueError: if the state is not in the network's state space
        """
        if state not in self:
            raise ValueError('state is not in state space')
        neighbors = [None] * self.size
        for i in range(self.size):
            neighbors[i] = copy.copy(state)
            neighbors[i][i] ^= 1
        return neighbors

    def distance(self, a, b):
        """
        Compute the Hamming distance between two states.

        .. rubric:: Examples

        .. doctest:: boolean_network

            >>> s_pombe.distance([0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0])
            4
            >>> s_pombe.distance([0, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0])
            0

        :param a: the first state
        :type a: list, numpy.ndarray
        :param b: the second state
        :type b: list, numpy.ndarray
        :return: the Hamming distance between the states
        :raises ValueError: if either state is not in the network's state space
        """
        if a not in self:
            raise ValueError('first state is not in state space')
        if b not in self:
            raise ValueError('second state is not in state space')
        out = 0
        for i in range(self.size):
            out += a[i] ^ b[i]
        return out


UniformNetwork.register(BooleanNetwork)
