# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
import numpy as np
from .statespace import StateSpace

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

    def state_space(self, n):
        """
        Return a :class:`StateSpace` object for a lattice of length ``n``.

        ::

            >>> eca = ECA(30)
            >>> eca.state_space(3)
            <neet.states.StateSpace object at 0x000001C0BDA38550>
            >>> space = eca.state_space(3)
            >>> list(space)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        :param n: the number of nodes in the lattice
        :type n: int
        :raises ValueError: if ``n < 1``
        """
        return StateSpace(n, base=2)

    def _unsafe_update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice``, in place, without
        checking the validity of the arguments.

        .. rubric:: Basic Use:

        ::

            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs)
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update([0,0,1,0,0])
            [1, 1, 1, 1, 1]

        .. rubric:: Single-Node Update:

        ::

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, index=1)
            [0, 1, 1, 0, 0]
            >>> xs
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update(xs, index=-1)
            [0, 1, 1, 0, 1]

        .. rubric:: State Pinning:

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, pin=[-2])
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca._unsafe_update(xs, pin=[4])
            [0, 1, 0, 1, 0]
        
        .. rubric:: Value Fixing:

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca._unsafe_update(xs, values={0:1,-2:0})
            [1, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> xs = [1,1,1,0,0]
            >>> ca._unsafe_update(xs, values={1:0,-1:0})
            [0, 0, 0, 1, 0]

        :param lattice: the one-dimensional sequence of states
        :type lattice: sequence
        :param index: the index to update (or None)
        :param pin: a sequence of indicies to pin (or None)
        :param values: a dictionary of index-value pairs to fix after update
        :returns: the updated lattice
        """
        pin_states = pin is not None and pin != []
        if self.boundary:
            left = self.__boundary[0]
            right = self.__boundary[1]
        else:
            left = lattice[-1]
            right = lattice[0]
        code = self.code
        if index is None:
            if pin_states:
                pinned = np.asarray(lattice)[pin]
            temp = 2 * left + lattice[0]
            for i in range(1, len(lattice)):
                temp = 7 & (2 * temp + lattice[i])
                lattice[i-1] = 1 & (code >> temp)
            temp = 7 & (2 * temp + right)
            lattice[-1] = 1 & (code >> temp)
            if pin_states:
                for (j, i) in enumerate(pin):
                    lattice[i] = pinned[j]
        else:
            if index < 0:
                index += len(lattice)

            if index == 0:
                temp = left
            else:
                temp = lattice[index-1]

            temp = 2 * temp + lattice[index]

            if index + 1 == len(lattice):
                temp = 2 * temp + right
            else:
                temp = 2 * temp + lattice[index+1]

            lattice[index] = 1 & (code >> (7 & temp))
        if values is not None:
            for key in values:
                lattice[key] = values[key]
        return lattice

    def update(self, lattice, index=None, pin=None, values=None):
        """
        Update the state of the ``lattice`` in place.

        .. rubric:: Basic Use:

        ::

            >>> ca = ECA(30)
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs)
            [0, 1, 1, 1, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update([0,0,1,0,0])
            [1, 1, 1, 1, 1]

        .. rubric:: Single-Node Update:

        ::

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, index=1)
            [0, 1, 1, 0, 0]
            >>> xs
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update(xs, index=-1)
            [0, 1, 1, 0, 1]

        .. rubric:: State Pinning:

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, pin=[-2])
            [0, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> ca.update(xs, pin=[4])
            [0, 1, 0, 1, 0]
        
        .. rubric:: Value Fixing:

            >>> ca.boundary = None
            >>> xs = [0,0,1,0,0]
            >>> ca.update(xs, values={0:1,-2:0})
            [1, 1, 1, 0, 0]
            >>> ca.boundary = (1,1)
            >>> xs = [1,1,1,0,0]
            >>> ca.update(xs, values={1:0,-1:0})
            [0, 0, 0, 1, 0]

        .. rubric:: Erroneous Usage:

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
            >>> ca.update(xs, index=5)
            Traceback (most recent call last):
                  ...
            IndexError: list index out of range
            >>> ca.update([0,0,1,0,0,], index=1, pin=[0])
                ...
            ValueError: cannot provide both the index and pin arguments
            >>> ca.update([0,0,1,0,0], index=1, values={0:0})
                ...
            ValueError: cannot provide both the index and values arguments
            >>> ca.update([0,0,1,0,0], pin=[2], values={2:0})
                ...
            ValueError: cannot set a value for a pinned state
            >>> ca.update([0,0,1,0,0], values={2:2})
                ...
            ValueError: invalid state in values argument

        :param lattice: the one-dimensional sequence of states
        :param index: the index to update (or None)
        :param pin: a sequence of indicies to pin (or None)
        :param values: a dictionary of index-value pairs to fix after update
        :returns: the updated lattice
        :raises ValueError: if ``lattice`` is not in the ECA's state space
        :raises IndexError: if ``index is not None and index > len(states)``
        :raises ValueError: if ``index`` and ``pin`` are both provided
        :raises ValueError: if ``index`` and ``values`` are both provided
        :raises ValueError: if an element of ``pin`` is a key in ``values``
        :raises ValueError: if a value in ``values`` is not binary (0 or 1)
        """
        size = len(lattice)
        if lattice not in self.state_space(size):
            raise ValueError("the provided state is not in the ECA's state space")

        if index is not None:
            if index < -size:
                raise IndexError("lattice index out of range")
            elif pin is not None and pin != []:
                raise ValueError("cannot provide both the index and pin arguments")
            elif values is not None and values != {}:
                raise ValueError("cannot provide both the index and values arguments")
        elif pin is not None and values is not None:
            for key in values.keys():
                for key in pin:
                    raise ValueError("cannot set a value for a pinned state")
        if values is not None:
            for val in values.values():
                if val != 0 and val != 1:
                    raise ValueError("invalid state in values argument")

        return self._unsafe_update(lattice, index, pin, values)

    def _incoming_neighbors_one_node(self,size,index):
        """
        Return the set of all neighbor nodes, where
        edge(neighbor_node-->index) exists.

        It is possible to call the neighbors of an index which is greater
        than the size of the network, in the case of networks which have
        fixed boundary conditions.

        The left boundary is at ``index==size+1``
        The right boundary is at ``index==size``

        eg. ``if size(eca)==3 and boundary!=None:``
        The organization of the neighbors list is as follows:
        ``[node_0|node_1|node_2|left_boundary|right_boundary]``

        :param size: size of ECA
        :param index: node index
        :returns: the set of all node indices which point toward the index node
        :raises ValueError: if ``index >= size and boundary==None``
        :raises ValueError: if ``index >= size+2 and boundary!=None``

        .. rubric:: Basic Use:

        ::

            >>> net = ECA(30)
            >>> net._incoming_neighbors_one_node(3,2)
            set([0,1,2])
            >>> net.boundary = (1,1)
            >>> net._incoming_neighbors_one_node(3,2)
            set([1,2,3])
            >>> net._incoming_neighbors_one_node(3,3)
            set([])
            >>> net._incoming_neighbors_one_node(3,4)
            set([])

        .. rubric:: Erroneous Usage:

        ::

            >>> net = ECA(30,boundary=(1,1))
            >>> net._incoming_neighbors_one_node(3,5)
            Traceback (most recent call last):
                ...
            ValueError: index must be <=(size+1) if self.boundary!=None
            >>> net.boundary = None
            >>> net._incoming_neighbors_one_node(3,3)
            Traceback (most recent call last):
                ...
            ValueError: index must be <size if self.boundary==None
        """
        if not isinstance(index, int):
            raise TypeError("Index must be a non-negative integer")

        elif index<0:
            raise ValueError("Index must be a non-negative integer")

        elif not isinstance(size, int):
            raise TypeError("Size must be a positive integer")

        elif size<1:
            raise ValueError("Size must be a positive integer")

        elif index>=size and self.boundary==None:
            raise ValueError("index must be <size if self.boundary==None")

        elif index>(size+1) and self.boundary!= None:
            raise ValueError("index must be <=(size+1) if self.boundary!=None")

        elif (index<=(size+1)) and (index>=size):
            return set([]) #bounary conditions have no incoming nodes

        elif (index<size) and (self.boundary == None):

            if size==1:

                left_neighbor = 0
                right_neighbor = 0

            else:

                if index==0:

                    left_neighbor = size-1
                    right_neighbor = 1

                elif index==(size-1):

                    left_neighbor = index-1
                    right_neighbor = 0

                else:

                    left_neighbor = index-1
                    right_neighbor = index+1

        elif (index<size) and (self.boundary!=None):

            if index==0:

                left_neighbor = size+1
                right_neighbor = 1

            else:

                left_neighbor = index-1
                right_neighbor = index+1

        return set([left_neighbor,index,right_neighbor])

    def _outgoing_neighbors_one_node(self,size,index):
        ## Allow index > size for calling boundary nodes?
        """
        Return the set of all neighbor nodes, where
        edge(index-->neighbor_node) exists.

        It is possible to call the neighbors of an index which is greater
        than the size of the network, in the case of networks which have
        fixed boundary conditions.

        The left boundary is at ``index==size+1``
        The right boundary is at ``index==size``

        eg. ``if size(eca)==3 and boundary!=None:``
        The organization of the neighbors list is as follows:
        ``[node_0|node_1|node_2|left_boundary|right_boundary]``

        :param size: size of ECA
        :param index: node index
        :returns: the set of all node indices which point toward the index node
        :raises ValueError: if ``index >= size and boundary==None``
        :raises ValueError: if ``index >= size+2 and boundary!=None``

        .. rubric:: Basic Use:

        ::

            >>> net = ECA(30)
            >>> net._outgoing_neighbors_one_node(3,2)
            set([0,1,2])
            >>> net.boundary = (1,1)
            >>> net._outgoing_neighbors_one_node(3,2)
            set([1,2])
            >>> net._outgoing_neighbors_one_node(3,3)
            set([2])
            >>> net._outgoing_neighbors_one_node(3,4)
            set([0])

        .. rubric:: Erroneous Usage:

        ::

            >>> net = ECA(30,boundary=(1,1))
            >>> net._outgoing_neighbors_one_node(3,5)
            Traceback (most recent call last):
                ...
            ValueError: index must be <=(size+1) if self.boundary!=None
            >>> net.boundary = None
            >>> net._outgoing_neighbors_one_node(3,3)
            Traceback (most recent call last):
                ...
            ValueError: index must be <size if self.boundary==None
        """
        if not isinstance(index, int):
            raise TypeError("Index must be a non-negative integer")

        elif index<0:
            raise ValueError("Index must be a non-negative integer")

        elif not isinstance(size, int):
            raise TypeError("Size must be a positive integer")

        elif size<1:
            raise ValueError("Size must be a positive integer")
        
        elif index>=size and self.boundary==None:
            raise ValueError("index must be <size if self.boundary==None")

        elif index>(size+1) and self.boundary!= None:
            raise ValueError("index must be <=(size+1) if self.boundary!=None")

        elif index==size:
            return set([size-1])

        elif index==(size+1):
            return set([0])

        elif (index<size) and (self.boundary == None):

            if size==1:

                neighbors = [0,0]

                # left_neighbor = 0
                # right_neighbor = 0

            else:

                if index==0:

                    neighbors = [size-1,1]

                    # left_neighbor = size-1
                    # right_neighbor = 1

                elif index==(size-1):

                    neighbors = [index-1,0]

                    # left_neighbor = index-1
                    # right_neighbor = 0

                else:

                    neighbors = [index-1,index+1]

                    # left_neighbor = index-1
                    # right_neighbor = index+1

        elif (index<size) and (self.boundary!=None):

            if size==1:

                neighbors = []

            else:

                if index==0:

                    neighbors = [1]

                    # left_neighbor = size+1
                    # right_neighbor = 1

                elif index==(size-1):

                    neighbors = [index-1]

                else:

                    neighbors = [index-1,index+1]

                    # left_neighbor = index-1
                    # right_neighbor = index+1

        return set(neighbors+[index])
        # return set([left_neighbor,index,right_neighbor])


    def neighbors(self,size,index=None,direction='both'):
        """
        Return a list of neighbors for each node.

        It is possible to call the neighbors of an index which is greater
        than the size of the network, in the case of networks which have
        fixed boundary conditions.

        The left boundary is at ``index==size+1``
        The right boundary is at ``index==size``

        eg. ``if size(eca)==3 and boundary!=None:``
        The organization of the neighbors list is as follows:
        ``[node_0|node_1|node_2|left_boundary|right_boundary]``
        
        :param size: size of ECA
        :param index: node index
        :param direction: type of node neighbors to return (can be 'in','out', or 'both')
        :returns: a set (if index!=None) or list of sets of neighbors of a node or network or nodes

        :param index: node index
        :returns: the set of all node indices which point toward the index node
        :raises ValueError: if ``index >= size and boundary==None``
        :raises ValueError: if ``index >= size+2 and boundary!=None``

        .. rubric:: Basic Use:

        ::

            >>> net = ECA(30)
            >>> net.neighbors(3,index=2,direction='out')
            set([0,1,2])
            >>> net.boundary = (1,1)
            >>> net.neighbors(3,index=2,direction='out')
            set([1,2])
            >>> net.neighbors(3,index=3,direction='out')
            set([2])
            >>> net.neighbors(3,index=4,direction='out')
            set([0])

        .. rubric:: Erroneous Usage:

        ::

            >>> net = ECA(30,boundary=(1,1))
            >>> net.neighbors(3,index=5,direction='out')
            Traceback (most recent call last):
                ...
            ValueError: index must be <=(size+1) if self.boundary!=None
            >>> net.boundary = None
            >>> net.neighbors(3,index=3,direction='out')
            Traceback (most recent call last):
                ...
            ValueError: index must be <size if self.boundary==None
        """
        if direction == 'in':
            if index:
                return self._incoming_neighbors_one_node(size,index)
            else:
                return [self._incoming_neighbors_one_node(size,node) for node in range(size)]

        if direction == 'out':
            if index:
                return self._outgoing_neighbors_one_node(size,index)
            else:
                return [self._outgoing_neighbors_one_node(size,node) for node in range(size)]

        if direction == 'both':
            if index:
                return self._incoming_neighbors_one_node(size,index)|self._outgoing_neighbors_one_node(size,index)
                       
            else:
                in_nodes = [self._incoming_neighbors_one_node(size,node) for node in range(size)]
                out_nodes = [self._outgoing_neighbors_one_node(size,node) for node in range(size)]
                return [in_nodes[i]|out_nodes[i] for i in range(size)]
