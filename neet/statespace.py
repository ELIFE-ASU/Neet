"""
.. currentmodule:: neet

.. testsetup:: statespace

    from neet import StateSpace, UniformSpace

The :mod:`neet` module provides the following classes from which all Neet
network classes inherit:

.. autosummary::
    :nosignatures:

    StateSpace
    UniformSpace

.. inheritance-diagram:: neet.StateSpace neet.UniformSpace
   :parts: 1

This endows networks with methods for iterating over the states of the network,
determining if a state exists in the network, and the ability to encode and
decode states as integer values. In other words, these classes provide an
interface for accessing the *unstructured* set of states of the network, with
no dynamical information.
"""
from .python import long


class StateSpace(object):
    """
    StateSpace represents a (potentially in-homogeneous) discrete state space.
    It implements iteration, inclusion testing and methods for encoding and
    decoding states as integers sutable for array indexing:

    .. autosummary::
       :nosignatures:

       size
       shape
       volume
       __iter__
       __contains__
       _unsafe_encode
       encode
       decode

    StateSpace instances are created from a ``shape`` array of integer
    representing the number of discrete states for each dimension of the state
    space.

    .. rubric:: Examples

    .. doctest:: statespace

        >>> StateSpace([2])      # 1-D state space
        <neet.statespace.StateSpace object at 0x...>
        >>> StateSpace([2,2])    # 2-D uniform state space
        <neet.statespace.StateSpace object at 0x...>
        >>> StateSpace([2,3,5])  # 3-D inhomogeneous space
        <neet.statespace.StateSpace object at 0x...>

    From the network perspective, each dimension of the state space corresponds
    to a node of the network. The number of discrete states of that node is the
    base of the corresponding dimension.

    The algorithms implemented by this class are intended to be as generic as
    possible. This comes at the cost of performance in some cases. This can be
    dealt with by deriving and overloading the appropriate methods, in
    particular :meth:`_unsafe_encode`. In fact, the following methods are
    recommended for overloading:

       * :meth:`__iter__`
       * :meth:`__contains__`
       * :meth:`_unsafe_encode`
       * :meth:`decode`

    The :meth:`encode` method uses :meth:`__contains__` and
    :meth:`_unsafe_encode` internally and rarely needs to be overloaded.

    :param shape: the base of each dimension of the state space
    :type shape: list
    :see: :class:`UniformSpace`
    """

    def __init__(self, shape):
        if isinstance(shape, list):
            if len(shape) == 0:
                raise ValueError("shape cannot be empty")
            else:
                self._volume = 1
                for base in shape:
                    if not isinstance(base, int):
                        raise TypeError("shape must be a list of ints")
                    elif base < 1:
                        raise ValueError("shape may only contain positive elements")
                    self._volume *= base
                self._size = len(shape)
                self._shape = shape[:]
        else:
            raise TypeError("shape must be a list")

    @property
    def size(self):
        """
        Get the size of the state space. That is the number of dimensions.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> StateSpace([2]).size
           1
           >>> StateSpace([2,3,4]).size
           3

        :returns: the number of dimensions of the state space
        """
        return self._size

    @property
    def shape(self):
        """
        Get the shape of the state space. That is the base of each dimension.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> StateSpace([2]).shape
           [2]
           >>> StateSpace([2,3,4]).shape
           [2, 3, 4]

        :returns: the shape of the state space
        """
        return self._shape

    @property
    def volume(self):
        """
        Get the volume of the state space. That is the number of states in the space.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> StateSpace([2]).volume
           2
           >>> StateSpace([2,3,4]).volume
           24

        :returns: the number of states in the space
        """
        return self._volume

    def __iter__(self):
        """
        Iterate over the states of the state space.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> list(StateSpace([2]))
           [[0], [1]]
           >>> list(StateSpace([2,2]))
           [[0, 0], [1, 0], [0, 1], [1, 1]]
           >>> list(StateSpace([3,2]))
           [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        """
        size, shape = self.size, self.shape
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            base = shape[i]
            if state[i] + 1 < base:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1

    def __contains__(self, states):
        """
        Determine if a state is in the state space.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> space = StateSpace([2])
           >>> [0] in space
           True
           >>> 0 in space
           False


        .. doctest:: statespace

           >>> space = StateSpace([3,2])
           >>> [2,0] in space
           True
           >>> [0,2] in space
           False
           >>> [2,0,0] in space
           False
        """
        try:
            if len(states) != self.size:
                return False

            for state, base in zip(states, self.shape):
                if state < 0 or state >= base:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        """
        Unsafely encode a state as an integer value.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> space._unsafe_encode([1,1])
           3

        The resulting numeric encodings must be consistent with the ordering of
        the states produced by :meth:`__iter__`. This allows necessary for
        memory-efficient implementations of many algorithms.

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> list(space)
           [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
           >>> list(map(space._unsafe_encode, space))
           [0, 1, 2, 3, 4, 5]

        .. Note::

            This method is **not** safe. It does not ensure that ``state`` is
            in fact in the space; if that's not the case then there are not
            guaruntees on the output. As such it should only be used in
            situations where the state is already known to be in the space,
            e.g. it is a state that was generated by :meth:`__iter__`. This is
            designed to allow algorithms to utilize state encoding without
            incurring the cost of consistency checking.

        :param state: the state as a list of coordinates
        :type state: int
        :returns: the state encoded as an integer

        :see: :meth:`encode`, :meth:`decode`
        """
        encoded, place = long(0), long(1)

        for (x, b) in zip(state, self.shape):
            encoded += place * long(x)
            place *= b

        return encoded

    def encode(self, state):
        """
        Encode a state as an integer.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> space.encode([1,1])
           3

        The resulting numeric encodings are consistent with the ordering of the
        states produced by :meth:`__iter__`.

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> list(space)
           [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
           >>> list(map(space.encode, space))
           [0, 1, 2, 3, 4, 5]


        This method is the inverse of the :meth:`decode` method:

        .. doctest:: statespace

           >>> space = StateSpace([3,2])
           >>> space.decode(space.encode([1,1]))
           [1, 1]
           >>> space.encode(space.decode(3))
           3

        :param state: the state as a list of coordinates
        :type state: int
        :returns: the state encoded as an integer

        :see: :meth:`encode`, :meth:`decode`
        """
        if state not in self:
            raise ValueError("state is not in state space")

        return self._unsafe_encode(state)

    def decode(self, encoded):
        """
        Decode an integer-encoded state into a coordinate list.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> space.decode(3)
           [1, 1]

        The resulting decoded states are consistent with the ordering of the
        states produced by :meth:`__iter__`.

        .. doctest:: statespace

           >>> space = StateSpace([2,3])
           >>> list(space)
           [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
           >>> list(map(space.decode, range(0,6)))
           [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]

        This method is the inverse of the :meth:`encode` method:

        .. doctest:: statespace

           >>> space = StateSpace([3,2])
           >>> space.decode(space.encode([1,1]))
           [1, 1]
           >>> space.encode(space.decode(3))
           3

        :param encoded: an integer-encoded state
        :type encoded: int
        :returns: the coordinate list of the decoded state

        :see: :meth:`encode`, :meth:`decode`
        """
        size = self.size
        state = [0] * size
        for (i, base) in enumerate(self.shape):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state


class UniformSpace(StateSpace):
    """
    A :class:`StateSpace` with the same number of states in each dimension.
    This allows for more efficient implementations of several methods.

    UniformSpace instances are created from their ``size`` and ``base``; the
    number of dimensions and the number of states in each dimension,
    respectively.

    In addition to the methods and attributes exposed by :class:`StateSpace`,
    the UniformSpace also provides:

    .. autosummary::
       :nosignatures:

       base

    .. rubric:: Examples

    .. doctest:: statespace

       >>> UniformSpace(1, 2) # 1-D unform space with base-2 dimensions
       <neet.statespace.UniformSpace object at 0x...>
       >>> UniformSpace(2, 2) # 2-D uniform space with base-2 dimensions
       <neet.statespace.UniformSpace object at 0x...>
       >>> UniformSpace(2, 4) # 2-D uniform space with base-4 dimension
       <neet.statespace.UniformSpace object at 0x...>

    :param size: the number of dimensions in the space
    :type size: int
    :param base: the number of states in each dimension
    :type base: int
    :see: :class:`StateSpace`
    """

    def __init__(self, size, base):
        super(UniformSpace, self).__init__([base] * size)
        self._base = base

    @property
    def base(self):
        """
        Get the base of the dimensions.

        .. rubric:: Examples

        .. doctest:: statespace

           >>> UniformSpace(2, 3).base
           3

        :returns: the base of the space's dimensions
        """
        return self._base

    def __iter__(self):
        size, base = self.size, self.base
        state = [0] * size
        yield state[:]
        i = 0
        while i != size:
            if state[i] + 1 < base:
                state[i] += 1
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

            base = self.base
            for x in state:
                if x < 0 or x >= base:
                    return False
            return True
        except TypeError:
            return False

    def _unsafe_encode(self, state):
        encoded, place = long(0), long(1)

        base = self.base
        for x in state:
            encoded += place * long(x)
            place *= base

        return encoded

    def decode(self, encoded):
        size, base = self.size, self.base
        state = [0] * size
        for i in range(size):
            state[i] = encoded % base
            encoded = int(encoded / base)
        return state
