# Copyright 2016-2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.


class StateSpace(object):
    """
    StateSpace represents the state space of a network model. It may be
    either uniform, i.e. all nodes have the same base, or non-uniform.
    """

    def __init__(self, spec, base=None):
        """
        Initialize the state spec in accordance with the provided ``spec``
        and base ``base``.

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
        :param base: the base of the network nodes (ignored if ``spec`` is a list)
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

            self.is_uniform = True
            self.ndim = spec
            self.base = base
            self.volume = base**spec

        elif isinstance(spec, list):
            if len(spec) == 0:
                raise ValueError("bases cannot be an empty")
            else:
                self.is_uniform = True
                self.volume = 1
                first_base = spec[0]
                if base is not None and first_base != base:
                    raise ValueError("base does not match base of spec")
                for spec_base in spec:
                    if not isinstance(spec_base, int):
                        raise TypeError("spec must be a list of ints")
                    elif spec_base < 1:
                        msg = "spec may only contain positive, nonzero elements"
                        raise ValueError(msg)
                    if self.is_uniform and spec_base != first_base:
                        self.is_uniform = False
                        if base is not None:
                            raise ValueError("b does not match base of spec")
                    self.volume *= spec_base
                self.ndim = len(spec)
                if self.is_uniform:
                    self.base = first_base
                else:
                    self.bases = spec[:]
        else:
            raise TypeError("spec must be an int or a list")

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
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
            [0, 1, 1], [1, 1, 1]]

        .. rubric:: Examples of Non-Boolean Spaces

        ::

            >>> list(StateSpace(1,b=3).states())
            [[0], [1], [2]]
            >>> list(StateSpace(2,b=4).states())
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1],
            [0, 2], [1, 2], [2, 2], [3, 2], [0, 3], [1, 3], [2, 3], [3, 3]]

        .. rubric:: Examples of Non-Uniform Spaces

        ::

            >>> list(StateSpace([1,2,3]).states())
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]
            >>> list(StateSpace([3,4]).states())
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2],
            [2, 2], [0, 3], [1, 3], [2, 3]]

        :yields: each possible state in the state space
        """
        state = [0] * self.ndim
        yield state[:]
        i = 0
        while i != self.ndim:
            base = self.base if self.is_uniform else self.bases[i]
            if state[i] + 1 < base:
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
            raise ValueError("state has the wrong length")
        encoded, place = 0, 1
        if self.is_uniform:
            base = self.base
            for i in range(self.ndim):
                if state[i] < 0 or state[i] >= base:
                    raise ValueError("invalid node state")
                encoded += place * state[i]
                place *= base
        else:
            for i in range(self.ndim):
                base = self.bases[i]
                if state[i] < 0 or state[i] >= base:
                    raise ValueError("invalid node state")
                encoded += place * state[i]
                place *= base
        return encoded

    def decode(self, encoded):
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

        :param encoded: the encoded state
        :type encoded: int
        :returns: the decoded state as a list
        """
        state = [0] * self.ndim
        if self.is_uniform:
            base = self.base
            for i in range(self.ndim):
                state[i] = encoded % base
                encoded = int(encoded / base)
        else:
            for i in range(self.ndim):
                base = self.bases[i]
                state[i] = encoded % base
                encoded = int(encoded / base)
        return state

    def check_states(self, states):
        """
        Check the validity of the provided states

        .. rubric:: Examples:

        ::

            >>> StateSpace(3).check_states([0,0,0])
            True
            >>> StateSpace(3).check_states([0,0])
            False
            >>> StateSpace(3).check_states([1,2,1])
            False

            >>> StateSpace([2, 3, 2]).check_states([0, 2, 1])
            True
            >>> StateSpace([2, 2, 3]).check_states([0, 1])
            False
            >>> StateSpace([2, 3, 4]).check_states([1, 1, 6])
            False

        :param states: the one-dimensional sequence of node states
        :returns: ``True`` if the ``states`` are valid
        """
        if len(states) != self.ndim:
            return False

        if self.is_uniform:
            for state in states:
                if self.base <= state or state < 0:
                    return False
        else:
            for state, base in zip(states, self.bases):
                if state not in range(base):
                    return False

        return True
