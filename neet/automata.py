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
        self.code = code
        self.boundary = boundary

    @property
    def code(self):
        """
        The Wolfram code of the elementary cellular automaton

        .. rubric:: Examples:

        ::

            >>> eca = ECA(30)
            >>> eca.code
            30
            >>> eca.code = 45
            >>> eca.code
            45
            >>> eca.code = 256
            >>> eca.code = 256
            Traceback (most recent call last):
                ...
            ValueError: invalid ECA code

        :type: int
        :raises TypeError: if ``code`` is not an instance of int
        :raises ValueError: if ``code`` is not in :math:`\{0,1,\ldots,255\}`
        """
        return self.__code

    @code.setter
    def code(self, code):
        if not isinstance(code, int):
            raise(TypeError("ECA code is not an int"))
        if 255 < code or code < 0:
            raise(ValueError("invalid ECA code"))
        self.__code = code

    @property
    def boundary(self):
        """
        The boundary conditions of the elemenary cellular automaton

        .. rubric:: Examples:

        ::

            >>> eca = ECA(30)
            >>> eca.boundary
            >>> eca.boundary = (0,1)
            >>> eca.boundary
            (0, 1)
            >>> eca.boundary = None
            >>> eca.boundary
            >>> eca.boundary = [0,1]
            Traceback (most recent call last):
                ...
            TypeError: ECA boundary are neither None nor a tuple

        :type: ``None`` or tuple
        :raises TypeError: if ``boundary`` is neither ``None`` or an instance of tuple
        :raises ValueError: if ``boundary`` is a neither ``None`` or a pair of binary states
        """
        return self.__boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary and not isinstance(boundary, tuple):
            raise(TypeError("ECA boundary are neither None nor a tuple"))
        if boundary:
            if len(boundary) != 2:
                raise(ValueError("invalid ECA boundary conditions"));
            for x in boundary:
                if x != 0 and x != 1:
                    raise(ValueError("invalid ECA boundary value"))
        self.__boundary = boundary

    @classmethod
    def check_lattice(self, lattice):
        """
        Check the validity of the provided lattice

        .. rubric:: Examples:

        ::

            >>> ECA.check_lattice([0])
            True
            >>> ECA.check_lattice([1,0])
            True
            >>> ECA.check_lattice([0,0,1])
            True

        ::

            >>> ECA.check_lattice([])
            Traceback (most recent call last):
                ...
            ValueError: lattice is empty
            >>> ECA.check_lattice([0,0,2])
            Traceback (most recent call last):
                ...
            ValueError: invalid value "2" in lattice
            >>> ECA.check_lattice(5)
            Traceback (most recent call last):
                ...
            TypeError: 'int' object is not iterable
            >>> ECA.check_lattice("elife")
            Traceback (most recent call last):
                ...
            ValueError: invalid value "e" in lattice

        :returns: ``True`` if the lattice is valid, otherwise an error is raised
        :raises ValueError: if ``not lattice``, i.e. if the lattice is empty
        :raises TypeError: if ``lattice`` is not iterable
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        if not lattice:
            raise(ValueError("lattice is empty"))

        for x in lattice:
            if x != 0 and x != 1:
                raise(ValueError("invalid value \"{}\" in lattice".format(x)))

        return True

    def _unsafe_update(self, lattice):
        """
        Update the state of the ``lattice``, in place, without
        checking the validity of the arguments.

        .. rubric:: Examples:

        ::

            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs)
            >>> xs
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (0,1)
            >>> ca._unsafe_update(xs)
            >>> xs
            [1, 1, 0, 0, 0]

        ::

            >>> xs = [0,0,2,0,0]
            >>> ca._unsafe_update(xs)
            >>> xs
            [0, 1, 1, 0, 1]

        :param lattice: the one-dimensional sequence of states
        :type lattice: sequence
        """
        if self.boundary:
            left  = self.__boundary[0]
            right = self.__boundary[1]
        else:
            left  = lattice[-1]
            right = lattice[0]

        d = 2 * left + lattice[0]
        for i in range(1, len(lattice)):
            d = 7 & (2 * d + lattice[i])
            lattice[i-1] = 1 & (self.code >> d)
        d = 7 & (2 * d + right)
        lattice[-1] = 1 & (self.code >> d)

    def update(self, lattice):
        """
        Update the state of the ``lattice`` in place.

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

        ::

            >>> xs = []
            >>> ca.update(xs)
            Traceback (most recent call last):
                ...
            ValueError: lattice is empty
            >>> xs = [0,0,2,0,0]
            >>> ca.update(xs)
            Traceback (most recent call last):
                ...
            ValueError: invalid value "2" in lattice

        :param lattice: the one-dimensional sequence of states
        :type lattice: sequence
        :raises ValueError: if ``not lattice``, i.e. if the lattice is empty
        :raises TypeError: if ``lattice`` is not iterable
        :raises ValueError: unless :math:`lattice[i] \in \{0,1\}` for all :math:`i`
        """
        ECA.check_lattice(lattice)
        self._unsafe_update(lattice)
