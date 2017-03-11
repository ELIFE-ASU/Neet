# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

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