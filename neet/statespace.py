"""
.. currentmodule:: neet.statespace

.. testsetup:: statespace

    from neet.statespace import StateSpace

State Space
===========

A fundamental property of networks is their state space. The
:class:`StateSpace` class provides a type for representing such a state
space. It has anumber of convenient properties, in particular

1. it is iterable, :meth:`StateSpace.__iter__`
2. you can ask if it contains a particular state,
   :meth:`StateSpace.__contains__`
3. it can :meth:`StateSpace.encode` states as integers, and
   :meth:`StateSpace.decode` integers into states

All of this together facilitates the clean implementation of many of the core
functions of :mod:`neet`.

API Documentation
-----------------
"""
from .python import long


class StateSpace(object):
    """
    :class:`StateSpace` represents the state space of a network model. It may be
    either uniform, i.e. all nodes have the same base, or non-uniform.
    """

    def __init__(self, spec, base=None):
        """
        Initialize the state spec in accordance with the provided ``spec``
        and base ``base``.

        .. rubric:: Examples of Uniform State Spaces:

        .. doctest:: statespace

            >>> spec = StateSpace(5)
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 5, 2)
            >>> spec = StateSpace(3, base=3)
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 3, 3)
            >>> spec = StateSpace([2, 2, 2])
            >>> (spec.is_uniform, spec.ndim, spec.base)
            (True, 3, 2)

        .. rubric:: Examples of Non-Uniform State Spaces:

        .. doctest:: statespace

            >>> spec = StateSpace([2, 3, 4])
            >>> (spec.is_uniform, spec.base, spec.ndim)
            (False, [2, 3, 4], 3)

        :param spec: the number of nodes or an array of node bases
        :type spec: int or list
        :param base: the base of the network nodes (ignored if ``spec`` is
                     a list)
        :raises TypeError: if ``spec`` is neither an int nor a list of ints
        :raises TypeError: if ``base`` is neither ``None`` nor an int
        :raises ValueError: if ``base`` is negative or zero
        :raises ValueError: if any element of ``spec`` is negative or zero
        :raises ValueError: if ``spec`` is empty
        """
        if isinstance(spec, int):
            if spec < 1:
                raise ValueError("ndim cannot be zero or negative")
            if base is None:
                base = 2
            elif not isinstance(base, int):
                raise TypeError("base must be an int")
            elif base < 1:
                raise ValueError("base must be positive, nonzero")

            self._is_uniform = True
            self._ndim = spec
            self._base = base
            self._volume = base**spec

        elif isinstance(spec, list):
            if len(spec) == 0:
                raise ValueError("bases cannot be an empty")
            else:
                self._is_uniform = True
                self._volume = 1
                first_base = spec[0]
                if base is not None and first_base != base:
                    raise ValueError("base does not match base of spec")
                for spec_base in spec:
                    if not isinstance(spec_base, int):
                        raise TypeError("spec must be a list of ints")
                    elif spec_base < 1:
                        msg = "spec may only contain positive elements"
                        raise ValueError(msg)
                    if self._is_uniform and spec_base != first_base:
                        self._is_uniform = False
                        if base is not None:
                            raise ValueError("b does not match base of spec")
                    self._volume *= spec_base
                self._ndim = len(spec)
                if self._is_uniform:
                    self._base = first_base
                else:
                    self._base = spec[:]
        else:
            raise TypeError("spec must be an int or a list")

    @property
    def ndim(self):
        """
        Get the dimensionality of the state space.
        """
        return self._ndim

    @property
    def base(self):
        """
        Get the base of each direction of the state space.

        If the state space is not uniform, the result is a list of bases
        (one for each dimension). Otherwise, the result is an integer.

        :return: a list of bases, one for each dimension
        """
        return self._base

    @property
    def volume(self):
        """
        Get the volume of the state space.
        """
        return self._volume

    @property
    def is_uniform(self):
        """
        Get whether every direction in the state space has the same base.
        """
        return self._is_uniform

    def __iter__(self):
        """
        Iterate over the states in the state space

        .. rubric:: Examples of Boolean Spaces

        .. doctest:: statespace

            >>> list(StateSpace(1))
            [[0], [1]]
            >>> list(StateSpace(2))
            [[0, 0], [1, 0], [0, 1], [1, 1]]
            >>> list(StateSpace(3))
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        .. rubric:: Examples of Non-Boolean Spaces

        .. doctest:: statespace

            >>> list(StateSpace(1, base=3))
            [[0], [1], [2]]
            >>> list(StateSpace(2, base=4))
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [2, 2], [3, 2], [0, 3], [1, 3], [2, 3], [3, 3]]

        .. rubric:: Examples of Non-Uniform Spaces

        .. doctest:: statespace

            >>> list(StateSpace([1,2,3]))
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]
            >>> list(StateSpace([3,4]))
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]]

        :yields: each possible state in the state space
        """
        state = [0] * self.ndim
        yield state[:]
        i = 0
        while i != self.ndim:
            base = self._base if self._is_uniform else self._base[i]
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
        Determine if a state is in the state space

        .. rubric:: Examples:

        .. doctest:: statespace

            >>> state_space = StateSpace(3)
            >>> [0, 0, 0] in state_space
            True
            >>> [0, 0] in state_space
            False
            >>> [1, 2, 1] in state_space
            False

        .. doctest:: statespace

            >>> [0, 2, 1] in StateSpace([2, 3, 2])
            True
            >>> [0, 1] in StateSpace([2, 2, 3])
            False
            >>> [1, 1, 6] in StateSpace([2, 3, 4])
            False

        :param states: the one-dimensional sequence of node states
        :returns: ``True`` if the ``states`` are valid
        """
        try:
            if len(states) != self.ndim:
                return False

            if self.is_uniform:
                for state in states:
                    if state not in range(self.base):
                        return False
            else:
                for state, base in zip(states, self.base):
                    if state not in range(base):
                        return False
        except TypeError:
            return False
        except IndexError:
            return False

        return True

    def _unsafe_encode(self, state):
        """
        Encode a state as an integer consistent with the state space, without
        checking the validity of the arguments. The encoding is such that,
        for example, the state [1, 0, 0] will correspond to 1; the state
        [1, 1, 0] will correspond to 3.

        .. rubric:: Examples:

        .. doctest:: statespace

            >>> space = StateSpace(3, base=2)
            >>> [space._unsafe_encode(s) for s in space]
            [0, 1, 2, 3, 4, 5, 6, 7]

        .. doctest:: statespace

            >>> space = StateSpace([2,3,4])
            >>> [space._unsafe_encode(s) for s in space]
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


        :param state: the state to encode
        :type state: list
        :returns: a unique integer encoding of the state
        :raises ValueError: if ``state`` has an incorrect length
        """
        encoded, place = long(0), long(1)

        base = self._base
        if self.is_uniform:
            for x in state:
                encoded += place * x
                place *= base
        else:
            for (x, b) in zip(state, base):
                encoded += place * x
                place *= b

        return long(encoded)

    def encode(self, state):
        """
        Encode a state as an integer consistent with the state space. The
        encoding is such that, for example, the state [1, 0, 0] will
        correspond to 1; the state [1, 1, 0] will correspond to 3.

        .. rubric:: Examples:

        .. doctest:: statespace

            >>> space = StateSpace(3, base=2)
            >>> [space.encode(s) for s in space]
            [0, 1, 2, 3, 4, 5, 6, 7]

        .. doctest:: statespace

            >>> space = StateSpace([2,3,4])
            >>> [space.encode(s) for s in space]
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


        :param state: the state to encode
        :type state: list
        :returns: a unique integer encoding of the state
        :raises ValueError: if ``state`` has an incorrect length
        """
        if state not in self:
            raise ValueError("state is not in state space")

        return self._unsafe_encode(state)

    def decode(self, encoded):
        """
        Decode an integer into a state in accordance with the state space.

        .. rubric:: Examples:

        .. doctest:: statespace

            >>> space = StateSpace(3)
            >>> list(space)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            >>> [space.decode(x) for x in range(space.volume)]
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        .. doctest:: statespace

            >>> space = StateSpace([2,3])
            >>> list(space)
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
            >>> [space.decode(x) for x in range(space.volume)]
            [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]

        :param encoded: the encoded state
        :type encoded: int
        :returns: the decoded state as a list
        """
        state = [0] * self._ndim
        base = self._base
        if self.is_uniform:
            b = base
            for i in range(self._ndim):
                state[i] = encoded % b
                encoded = int(encoded / b)
        else:
            for i in range(self._ndim):
                b = base[i]
                state[i] = encoded % b
                encoded = int(encoded / b)
        return state
