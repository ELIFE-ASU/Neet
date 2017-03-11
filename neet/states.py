# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

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
            self.base  = b

        elif isinstance(spec, list):
            if len(spec) == 0:
                raise(ValueError("bases cannot be an empty"))
            else:
                self.is_uniform = True
                b = spec[0]
                for x in spec:
                    if not isinstance(x, int):
                        raise(TypeError("spec must be a list of ints"))
                    elif x < 1:
                        raise(ValueError("spec may only contain positive, nonzero elements"))
                    if x != b:
                        self.is_uniform = False
                self.ndim = len(spec)
                if self.is_uniform:
                    self.base  = b
                else:
                    self.bases = spec[:]
        else:
            raise(TypeError("spec must be an int or a list"))

def states(spec, b=2):
    """
    Generate all possible network states according to some specification,
    ``spec``.

    If ``spec`` is an integer, then it is taken to be the number of nodes in a
    network and ``b`` is assumed to be the base of the all of the nodes. As
    such, this function generates all base-``b`` sequences of length ``spec``.

    .. rubric:: Example:

    ::

        >>> list(neet.states(1))
        [[0], [1]]
        >>> list(neet.states(2))
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> list(neet.states(3))
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1
        , 0, 1], [0, 1, 1], [1, 1, 1]]

    ::

        >>> list(neet.states(2, b=1))
        [[0, 0]]
        >>> list(neet.states(2, b=2))
        [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> list(neet.states(2, b=3))
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1
        , 2], [2, 2]]

    If, however, ``spec`` is a list of integers, then each integer is assumed to
    be the base of some node in the network. The second second argument ``b``
    is ignored. In this case, it generates all sequences of length ``len(spec)``
    where the base of element ``i`` is ``spec[i]``.

    .. rubric:: Example:

    ::

        >>> list(neet.states([]))
        [[]]
        >>> list(neet.states([1]))
        [[0]]
        >>> list(neet.states([2]))
        [[0], [1]]
        >>> list(neet.states([3]))
        [[0], [1], [2]]
        >>> list(neet.states([2,3]))
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
        >>> list(neet.states([3,3]))
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2]
        , [1, 2], [2, 2]]

    :param spec: the number of nodes or an array of node bases
    :type spec: int or list
    :param b: the base of the network nodes (ignored is ``spec`` if an list)
    :yields: a possible network state
    :raises TypeError: if ``spec`` is neither an int nor a list of ints
    :raises ValueError: if ``b`` is negative or zero
    :raises ValueError: if any element of ``spec`` is negative or zero
    """
    if isinstance(spec, int):
        if not isinstance(b, int):
            raise(TypeError("base must be an int"))
        elif b < 1:
            raise(ValueError("base must be positive, nonzero"))

        for state in states([b]*spec):
            yield state
    else:
        for i in range(len(spec)):
            if not isinstance(spec[i], int):
                raise(TypeError("spec is not an int nor a list of ints"))
            elif spec[i] < 1:
                raise(ValueError("spec has a zero or negative base"))
        n = len(spec)
        state = [0]*n
        yield state[:]
        i = 0
        while i != n:
            if state[i] + 1 < spec[i]:
                state[i] += 1
                for j in range(i):
                    state[j] = 0
                i = 0
                yield state[:]
            else:
                i += 1