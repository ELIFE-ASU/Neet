# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.

class ECA(object):
    """
    ECA is a class to represent elementary cellular automaton rules. Each ECA
    contains an 8-bit integral member variable ``code`` representing the
    Wolfram code for the ECA rule and a set of boundary conditions which is
    either ``None``, signifying periodic boundary conditions, or a pair of
    cell states signifying fixed, open boundary conditions.
    """

    def __init__(self, code, boundary=None):
        """
        Construct an elementary cellular automaton rule.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> ca.code
            30
            >>> ca.boundary
            >>> ca = ECA(30, boundary=(0,0))
            >>> ca.boundary
            (0,0)
            
        :param code: the Wolfram code for the ECA
        :type code: int
        :param boundary: the boundary conditions for the CA
        :type boundary: tuple or None
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\{0,1,\ldots,255\}`
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of binary states
        """
        if not isinstance(code, int):
            raise(TypeError("ECA code is not an int"))
        if boundary and not isinstance(boundary, tuple):
            raise(TypeError("ECA boundary are neither None nor a tuple"))
        if 255 < code or code < 0:
            raise(ValueError("invalid ECA code"))
        if boundary:
            if len(boundary) != 2:
                raise(ValueError("invalid ECA boundary conditions"));
            for x in boundary:
                if x != 0 and x != 1:
                    raise(ValueError("invalid ECA boundary value"))
        self.code = code
        self.boundary = boundary

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

    def __unsafe_update_closed(self, lattice, n):
        """
        Update the state of the ``lattice`` ``n``-times, in place, without
        checking the validity of the arguments. This method uses closed
        (a.k.a. periodic or cyclic) boundary conditions.
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        """
        for m in range(n):
            a = lattice[0]            
            d = 2 * lattice[-1] + lattice[0]
            for i in range(1, len(lattice)):
                d = 7 & (2 * d + lattice[i])
                lattice[i-1] = 1 & (self.code >> d)
            d = 7 & (2 * d + a)
            lattice[-1] = 1 & (self.code >> d)
            
    def __unsafe_update_open(self, lattice, n):
        """
        Update the state of the ``lattice`` ``n``-times, in place, without
        checking the validity of the arguments. This method uses fixed,
        open boundary conditions.
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        """
        for m in range(n):
            d = 2 * self.boundary[0] + lattice[0]
            for i in range(1, len(lattice)):
                d = 7 & (2 * d + lattice[i])
                lattice[i-1] = 1 & (self.code >> d)
            d = 7 & (2 * d + self.boundary[1])
            lattice[-1] = 1 & (self.code >> d)

    def update(self, lattice, n=1):
        """
        Update the state of the ``lattice`` ``n``-times in place.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs)
            >>> xs
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (0,1)
            >>> ca.update(xs)
            >>> xs
            [1, 1, 0, 0, 0]
            >>> ca.update(xs, n=2)
            >>> xs
            [1, 0, 1, 0, 1]
        
        :param lattice: the one-dimensional sequence of states
        :type lattice: seq
        :param n: the number of times to update the state
        :type n: int
        :raises ValueError: if ``n`` is less than 1
        :raises ValueError: if :math:`\|lattice\| < 3`
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        ECA.__check_arguments(lattice, n)
        if self.boundary:
            self.__unsafe_update_open(lattice, n)
        else:
            self.__unsafe_update_closed(lattice, n)

    def step(self, lattice, n=1):
        """
        Update a copy of the state of the ``lattice`` ``n``-times.
        
        .. rubric:: Examples:
        
        ::
        
            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca.step(xs)
            [0, 1, 1, 1, 0]
            >>> xs
            [0, 0, 1, 0, 0]
            >>> ca.boundary = (0,1)
            >>> ca.step(xs)
            [0, 1, 1, 1, 1]
            >>> ca.step(xs, n=2)
            [1, 1, 0, 0, 0]
                    
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
        if self.boundary:
            self.__unsafe_update_open(l, n)
        else:
            self.__unsafe_update_closed(l, n)
        return l
