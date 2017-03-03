# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

class ECA(object):
    """
    ECA is a class to represent elementary cellular automaton rules. Each ECA
    contains an 8-bit integral member variable ``code`` representing the
    Wolfram code for the ECA rule.
    """

    def __init__(self, code):
        """
        Construct an elementary cellular automaton rule.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> ca.code
            30
            
        :param code: the Wolfram code for the ECA
        :type code: int
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\{0,1,\ldots,255\}`.
        """
        if not isinstance(code, int):
            raise(TypeError("ECA code is not an int"))
        if 255 < code or code < 0:
            raise(ValueError("invalid ECA code"))
        self.code = code

    @classmethod
    def __check_arguments(self, lattice, n):
        """
        Check the validity of the provided arguments.
        
        :raises ValueError: if ``n`` is less than 1
        :raises ValueError: if :math:`\|lattice\| < 3`
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        if n < 1:
            raise(ValueError("cannot update lattice fewer than once"))

        if len(lattice) < 3:
            raise(ValueError("lattice is too short"))

        for x in lattice:
            if x != 0 and x != 1:
                msg = "invalid value {} in lattice".format(x)
                raise(ValueError(msg))

    def __unsafe_update(self, lattice, n):
        """
        Update the state of the ``lattice`` ``n``-times, in place, without
        checking the validity of the arguments.
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        """
        for m in range(n):
            a = lattice[0]
            d = 2 * lattice[-1] + lattice[0]
            for i in range(1,len(lattice)):
                d = 7 & (2 * d + lattice[i])
                lattice[i-1] = 1 & (self.code >> d)
            d = 7 & (2 * d + a)
            lattice[-1] = 1 & (self.code >> d)

    def update(self, lattice, n=1):
        """
        Update the state of the ``lattice`` ``n``-times in place.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> xs = [0,1,0]
            >>> ca.update(xs)
            >>> xs
            [1, 1, 1]
            >>> ca.update(xs)
            >>> xs
            [0, 0, 0]
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, n=2)
            >>> xs
            [1, 1, 0, 0, 1]
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        :raises ValueError: if ``n`` is less than 1
        :raises ValueError: if :math:`\|lattice\| < 3`
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        ECA.__check_arguments(lattice, n)
        self.__unsafe_update(lattice, n)

    def step(self, lattice, n=1):
        """
        Update a copy of the state of the ``lattice`` ``n``-times.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> ca.step(xs)
            [1, 1, 1]
            >>> xs
            [0, 1, 0]
            >>> ca.step(xs, n=2)
            [0, 0, 0]
            >>> xs = [0,0,1,0,0]
            >>> ca.step(xs, n=2)
            [1, 1, 0, 0, 1]
            >>> xs
            [0, 0, 1, 0, 0]
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        :returns: an updated copy of ``lattice``
        :raises ValueError: if ``n`` is less than 1
        :raises ValueError: if :math:`\|lattice\| < 3`
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        ECA.__check_arguments(lattice, n)
        l = lattice[:]
        self.__unsafe_update(l, n)
        return l
