"""
.. currentmodule:: neet

.. testsetup:: landscape

    from neet.boolean import ECA
    from neet.boolean.examples import s_pombe
    import numpy as np

The :mod:`neet` module provides the :class:`LandscapeMixin` class from which
the :class:`neet.Network` class inherits. This endows all networks with the
various methods for computing the various landscape-related properties of the
networks, such as :attr:`LandscapeMixin.attractors`. These properties are often
associated with the *state space* of the network; however, we have opted to
provide them via a separate mixin because the :class:`neet.StateSpace` class
represents an *unstructured* set of states, with no dynamical information

A key feature of the :class:`LandscapeMixin` is that it is lazy and caches
results as they are computed. For example, the attractors of the landscape are
computed the first the user requests the :attr:`LandscapeMixin.attractors`
property, but the result is cached in the :attr:`LandscapeMixin.landscape_data`
attribute. Subsequent calls simply return the cached data. What's more, many of
the properties of the landscape can be determined using almost the exact same
algorithm, so whenever one is requested, they are all simultaneously computed.
See :class:`LandscapeMixin.expound` for a list of such properties.
"""
import networkx as nx
import numpy as np
import pyinform as pi


class LandscapeData(object):
    """
    The LandscapeData class stores the various landscape properties computed in
    the :class:`LandscapeMixin`. This is used rather an individual properties
    within :class:`LandscapeMixin` to make it simple for users to extract all
    of the landscape properties before modifying a network and observing the
    effects of that change on the landscape.

    The following properties are stored in LandscapeData:

    .. autosummary::
       :nosignatures:

        LandscapeMixin.transitions
        LandscapeMixin.attractors
        LandscapeMixin.attractor_lengths
        LandscapeMixin.basins
        LandscapeMixin.basin_sizes
        LandscapeMixin.basin_entropy
        LandscapeMixin.heights
        LandscapeMixin.recurrence_times
        LandscapeMixin.in_degrees

    .. rubric:: Basic Usage

    .. doctest:: landscape

        >>> s_pombe.attractors
        array([array([76]), array([4]), array([8]), array([12]),
               array([144, 110, 384]), array([68]), array([72]), array([132]),
               array([136]), array([140]), array([196]), array([200]),
               array([204])], dtype=object)
        >>> default_landscape = s_pombe.landscape_data

        >>> s_pombe.landscape(pin=[0,1]).attractors
        array([array([0]), array([1]), array([386, 402, 178, 162]),
               array([387, 403, 179, 163]), array([4]), array([8]), array([12]),
               array([76]), array([65]), array([64]), array([68]), array([72]),
               array([132]), array([136]), array([140]), array([192]),
               array([193]), array([196]), array([200]), array([204])],
              dtype=object)

        >>> default_landscape.attractors
        array([array([76]), array([4]), array([8]), array([12]),
               array([144, 110, 384]), array([68]), array([72]), array([132]),
               array([136]), array([140]), array([196]), array([200]),
               array([204])], dtype=object)

        >>> s_pombe.clear_landscape()
    """
    transitions = None
    attractors = None
    attractor_lengths = None
    basins = None
    basin_sizes = None
    basin_entropy = None
    heights = None
    recurrence_times = None
    in_degrees = None


class LandscapeMixin:
    """
    The LandscapeMixin class represents the structure and topology of the
    "landscape" of state transitions. That is, it is the state space together
    with information about state transitions and the topology of the state
    transition graph.

    The LandscapeMixin class exposes the following methods:

    .. autosummary::
       :nosignatures:

       landscape
       clear_landscape
       landscape_data
       transitions
       attractors
       attractor_lengths
       basins
       basin_sizes
       basin_entropy
       heights
       recurrence_times
       in_degrees
       trajectory
       timeseries
       landscape_graph
       draw_landscape_graph
       expound
    """

    # Whether or not the landscape data has been populated
    __landscaped = False
    # The landscape data cache
    __landscape_data = LandscapeData()

    def landscape(self, index=None, pin=None, values=None):
        """
        Setup the landscape.

        Prepares the landscape for computation of the various properties,
        specifying which nodes will be updated (``index``), pinned (``pin``) or
        set to a particular state (``values``). In particular, it computes the
        state transitions of the network and prepares private variables for a
        subsequent call to :meth:`expound`, :meth:`landscape_graph`, etc...

        This function is implicitly called with no arguments by the various
        landscape accessors if it has not already been called. This is intended
        as a convenience since most of the time the user would do this anyway.

        This function implicitly calls :attr:`clear_landscape`, so make sure to
        create a reference to :attr:`landscape_data` if landscape information
        has previously been compute and you wish to keep it around.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.landscape_data.transitions
            >>> s_pombe.landscape()
            <neet.boolean.wtnetwork.WTNetwork object at 0x...>
            >>> len(s_pombe.landscape_data.transitions)
            512

        .. rubric:: Pinning States

        .. doctest:: landscape

            # Prevents all states from transitioning
            >>> s_pombe.landscape(pin = range(s_pombe.size))
            <neet.boolean.wtnetwork.WTNetwork object at 0x...>
            >>> np.array_equal(s_pombe.landscape_data.transitions, range(s_pombe.volume))
            True
            >>> s_pombe.clear_landscape()

        .. rubric:: Overriding Node States

        .. doctest:: landscape

            # Forces all states to transition to 0
            >>> s_pombe.landscape(values={i: 0 for i in range(s_pombe.size)})
            <neet.boolean.wtnetwork.WTNetwork object at 0x...>
            >>> np.all(s_pombe.landscape_data.transitions == 0)
            True
            >>> s_pombe.clear_landscape()

        :param index: the index to update (or None)
        :param pin: the indices to pin during update (or None)
        :param values: a dictionary of index-value pairs to set after update
        :return: ``self``
        """

        self.__index = index
        self.__pin = pin
        self.__values = values

        self.__expounded = False

        update = self._unsafe_update
        encode = self._unsafe_encode

        transitions = np.empty(self.volume, dtype=np.int)
        for i, state in enumerate(self):
            transitions[i] = encode(update(state,
                                           index=self.__index,
                                           pin=self.__pin,
                                           values=self.__values))

        self.clear_landscape()
        self.__landscape_data.transitions = transitions
        self.__landscaped = True

        return self

    def clear_landscape(self):
        """
        Clear the landscape's data and graph from memory.
        """
        self.__landscaped = False
        self.__landscape_graph = None
        self.__landscape_data = LandscapeData()

    @property
    def landscape_data(self):
        """
        Get the :class:`LandscapeData` object.

        The :class:`LandscapeData` object contains any cached attractor
        landscape information generated by a call to :meth:`expound`.
        """
        return self.__landscape_data

    @property
    def transitions(self):
        """
        Get the state transitions as an array. Each element of the array is
        the next (encoded) state of the system starting from the initial state
        equal to the index. For example, if

        ::

            >>> net.transitions
            array([ 0, 3, 1, 2 ])

        then state ``0`` will transition to ``0``, ``1`` to ``3``, etc... Be
        aware that if :meth:`landscape` has not been called, this method will
        call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.transitions
            array([  2,   2, 130, 130,   4,   0, 128, 128,   8,   0, 128, 128,  12,
                     0, 128, 128, 256, 256, 384, 384, 260, 256, 384, 384, 264, 256,
                   ...
                   208, 208, 336, 336, 464, 464, 340, 336, 464, 464, 344, 336, 464,
                   464, 348, 336, 464, 464])

        .. rubric:: Pinned States

        A preceding call to :meth:`landscape` can, for example, pin specific
        nodes to their current state, thus affecting the state transitions.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin = [0]).transitions
            array([  2,   3, 130, 131,   4,   1, 128, 129,   8,   1, 128, 129,  12,
                     1, 128, 129, 256, 257, 384, 385, 260, 257, 384, 385, 264, 257,
                   ...
                   208, 209, 336, 337, 464, 465, 340, 337, 464, 465, 344, 337, 464,
                   465, 348, 337, 464, 465])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of state transitions
        """
        if not self.__landscaped:
            self.landscape()
        return self.__landscape_data.transitions

    @property
    def attractors(self):
        """
        Get the attractors of the landscape as an array. Each element of the
        array is an attractor cycle, each of which is an array of states in
        the cycle. If :meth:`landscape` has not been called, this method will
        implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.attractors
            array([array([76]), array([4]), array([8]), array([12]),
                   array([144, 110, 384]), array([68]), array([72]), array([132]),
                   array([136]), array([140]), array([196]), array([200]),
                   array([204])], dtype=object)

        .. rubric:: Update Only a Single Node

        A preceding call to :meth:`landscape` can, for example, specify which
        nodes will be updated in the process of computing the attractors. For
        example, we can allow only the first node of the state to be updated.

        .. doctest:: landscape

            >>> s_pombe.landscape(index=0).attractors
            array([[  0],
                   [  2],
                   [  4],
                  ...
                   [506],
                   [508],
                   [510]])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of attractor cycles, each of which is
                  an array of encoded states
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.attractors

    @property
    def attractor_lengths(self):
        """
        Get the length of the attractors as an array. The array is indexed by
        the basin number. The order of the attractor lengths is the same as in
        :attr:`attractors`. For example,

        ::

            >>> net.attractors
            array([ array([0,1]), array([1]) ]
            >>> net.attractor_lengths
            array([2, 1])

        If :meth:`landscape` has not been called, this method will implicitly
        call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.attractor_lengths
            array([1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1])

        .. rubric:: Pinned States

        A preceding call to :meth:`landscape` can pin specific nodes to their
        current state, thus affecting the attractor lengths.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin = [0]).attractor_lengths
            array([1, 6, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of the lengths of the attractors
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.attractor_lengths

    @property
    def basins(self):
        """
        Get the basins of the states as an array. Each index of the array is
        an encoded state and the corresponding value is the attractor basin in
        which it resides. The attractor basins are integers which can be used
        to index the :attr:`attractors` array, providing the attractor cycle
        for the base. For example, if

        ::

            >>> net.basins
            array([ 0, 1, 2, 1 ])
            >>> net.attractors
            array([ array([0]), array([1]), array([2]) ])

        then the states ``1`` and ``3`` are both in the attractor basin which
        attracts to the fixed-point ``1``. If :meth:`landscape` has not been
        called, this method will implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.basins
            array([ 0,  0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  0,  3,  0,  0,  0,  0,
                    0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  4,  4,
                    ...
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0])

        .. rubric:: Resetting Node States

        A preceding call to :meth:`landscape` can, for example, specify that
        specific nodes are reset to a particular value after the updating
        the. For example, we can force the first and second nodes to ``0``,
        thus affecting the basins.

        .. doctest:: landscape

            >>> s_pombe.landscape(values={0: 0, 1: 0}).basins
            array([ 0,  0,  1,  1,  2,  0,  1,  1,  3,  0,  1,  1,  4,  0,  1,  1,  1,
                    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                  ...
                    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                    1,  1])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of each state's attractor basin
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.basins

    @property
    def basin_sizes(self):
        """
        Get the sizes of the attractor basins as an array. The array is indexed
        by the basin number. The order of the basin sizes is the same as in
        :attr:`attractors`. For example, if

        ::

            >>> net.attractors
            array([ array([0,1]), array([3,6]) ]
            >>> net.basin_sizes
            array([ 5, 3 ])

        then the attractor ``[0, 1]`` has a basin size of :math:`5` with the
        remaining states in the other attractor's basin. If :meth:`landscape`
        has not been called, this method will implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.basin_sizes
            array([378,   2,   2,   2, 104,   6,   6,   2,   2,   2,   2,   2,   2])

        .. rubric:: Pinning States

        A preceding call to :meth:`landscape` can specify that some of the
        nodes are not updated, say the first two.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin=[0,1]).basin_sizes
            array([  1,   4, 128, 128,   1,   1,   1, 114, 120,   1,   1,   1,   1,
                     1,   1,   1,   4,   1,   1,   1])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of each attractor's basin size
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.basin_sizes

    @property
    def basin_entropy(self):
        """
        Compute the basin entropy of the landscape [Krawitz2007]_. That is
        the Shannon entropy (in bits) of the distribution of basin sizes. For
        example,

        ::

            >>> net.basin_sizes
            array([6, 2])
            >>> net.basin_entropy
            0.8112781244591328

        which is :math:`-\\frac{6}{8}\\log_2{\\frac{6}{8}) -
        \\frac{2}{8}\\log_2{\\frac{2}{8})`. If :meth:`landscape` has not
        been called, this method will implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.basin_entropy
            1.2218888...

        .. rubric:: Pinning States

        A preceding call to :meth:`landscape` can specify that some of the
        nodes are not updated, say the first two.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin=[0,1]).basin_entropy
            2.328561849437885
            >>> s_pombe.clear_landscape()

        :return: basin entropy in bits
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()

        return self.__landscape_data.basin_entropy

    @property
    def heights(self):
        """
        Get the heights of each state in the landscape. That is the fewest
        number of time steps from that state to a state in it's attractor
        cycle, as an array. Each index of the array is an encoded state, and
        the corresponding value is the height. For example, if

        ::

            >>> net.heights
            array([ 3, 0, 1, ... ])

        then it will take :math:`3` time steps for the state ``0`` to reach
        an attractor state while state ``1`` **is** an attractor state`. If
        :meth:`landscape` has not been called, this method will implicitly
        call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.heights
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 1, 1, 2, 8,
                   1, 1, 2, 8, 1, 1, 2, 8, 1, 1, 2, 2, 2, 2, 9, 9, 1, 1, 9, 9, 1, 1,
                   ...
                   3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3])

        .. rubric:: Resetting Node States

        A preceding call to :meth:`landscape` can specify that
        specific nodes are reset to a particular value after the updating
        the. For example, we can force the first and second nodes to ``0``,
        thus affecting the basins.

        .. doctest:: landscape

            >>> s_pombe.landscape(values={0: 0, 1: 0}).heights
            array([0, 1, 6, 6, 0, 1, 6, 6, 0, 1, 6, 6, 0, 1, 6, 6, 2, 2, 5, 5, 2, 2,
                   5, 5, 2, 2, 5, 5, 2, 2, 5, 5, 3, 3, 6, 6, 3, 3, 6, 6, 3, 3, 6, 6,
                  ...
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3])
            >>> s_pombe.clear_landscape()


        :return: a :class:`numpy.ndarray`, each value of which is the height of the
                 indexing state
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.heights

    @property
    def recurrence_times(self):
        """
        Get the recurrence time of each state in the landscape. That is the
        number of time steps from that state after which *some* state is
        repeated, as an array. Each index of the array is an encoded state,
        and the corresponding value is the recurrence time of that state. For
        example, if

        ::

            >>> net.recurrent_times
            array([ 3, 10, 0, ... ])

        then a state will be seen at least twice if the ``0`` state is updated
        more than :math:`3` times. The ``2`` state is a fixed-point attractor
        state as updating even once will repeat a state. If :meth:`landscape`
        has not been called, this method will implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.recurrence_times
            array([7, 7, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 0, 8, 6, 6, 8, 8, 3, 3, 2, 8,
                   3, 3, 2, 8, 3, 3, 2, 8, 3, 3, 4, 4, 4, 4, 9, 9, 3, 3, 9, 9, 3, 3,
                   ...
                   3, 9, 9, 9, 3, 9, 9, 9, 3, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3])

        .. rubric:: Resetting Node States

        A preceding call to :meth:`landscape` can specify that
        specific nodes are reset to a particular value after the updating
        the. For example, we can force the first and second nodes to ``0``,
        thus affecting the basins.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin=[0,1]).recurrence_times
            array([0, 0, 5, 5, 0, 1, 5, 5, 0, 1, 5, 5, 0, 1, 5, 5, 2, 2, 4, 4, 2, 2,
                   4, 4, 2, 2, 4, 4, 2, 2, 4, 4, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 5, 5,
                   ...
                   3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 5, 5, 3, 3, 8, 8, 3, 3, 8, 8, 3, 3,
                   8, 8, 3, 3, 8, 8])
            >>> s_pombe.clear_landscape()


        :return: a :class:`numpy.ndarray` of recurrence times, one for each state
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.recurrence_times

    @property
    def in_degrees(self):
        """
        Get the in-degree of each state in the landscape. That is the number
        of states which transition to that state in a single time step,
        as a array. Each index of the array is an encoded state, and the
        corresponding value is the number of preceding states. For example, if

        ::

            >>> net.in_degrees
            array([ 5, 2, 0, 0, ... ]

        then :math:`5` states transition to the ``0`` state in a single
        time step, while states ``2`` and ``3`` are in the `Garden of Eden
        <https://wikipedia.org/wiki/Garden_of_Eden_(cellular_automaton)>`_. If
        :meth:`landscape` has not been called, this method will implicitly
        call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.in_degrees
            array([ 6,  0,  4,  0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  0, 12,
                    0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    ...
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0])

        .. rubric:: Pinning States

        A preceding call to :meth:`landscape` can specify that some of the
        nodes are not updated, say nodes ``7`` and ``8``.

        .. doctest:: landscape

            >>> s_pombe.landscape(pin=[7,8]).in_degrees
            array([36,  0,  6,  0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  0, 42,
                    0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  ...
                    0,  1,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0])
            >>> s_pombe.clear_landscape()

        :return: a :class:`numpy.ndarray` of the in-degree of each state
        """
        if not self.__landscaped:
            self.landscape()
        if not self.__expounded:
            self.expound()
        return self.__landscape_data.in_degrees

    def landscape_graph(self, **kwargs):
        """
        Construct a :class:`networkx.DiGraph` of the state transitions.

        If :meth:`landscape` has not been called, this method will implicitly call it.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.landscape_graph()
            <networkx.classes.digraph.DiGraph object at 0x...>

        :param kwargs: kwargs to pass to :class:`networkx.DiGraph`
        :return: a :class:`networkx.DiGraph` representing the state transition
                 graph of the landscape
        """
        if not self.__landscaped:
            self.landscape()
        if self.__landscape_graph is None:
            self.__landscape_graph = nx.DiGraph(
                list(enumerate(self.__landscape_data.transitions)), **kwargs)
        elif (len(kwargs) != 0):
            self.__landscape_graph.graph.update(kwargs)
        return self.__landscape_graph

    def draw_landscape_graph(self, graphkwargs={}, pygraphkwargs={}):
        """
        Draw the state transition graph.

        This method requires the optional dependency `pygraphviz
        <https://pygraphviz.github.io>`_, which can be installed via
        ``pip``. Be aware that ``pygraphviz`` requires native binaries of
        `Graphviz <https://graphviz.org>`_ which **cannot** be installed via
        pip.

        If :meth:`landscape` has not been called, this method will implicitly call it.

        .. rubric:: Basic Usage

        ::

            >>> s_pombe.draw_landscape_graph()

        :param graphkwargs: kwargs to pass to `landscape_graph`
        :param pygraphkwargs: kwargs to pass to `view_pygraphviz`
        """
        from .draw import view_pygraphviz
        default_args = {'prog': 'dot'}
        graph = self.landscape_graph(**graphkwargs)
        view_pygraphviz(graph, **dict(default_args, **pygraphkwargs))

    def trajectory(self, init, timesteps=None, encode=None):
        """
        Compute the trajectory from a given state.

        This method computes a trajectory from ``init`` to the last before
        the trajectory begins to repeat. If ``timesteps`` is provided, then
        the trajectory will have a length of ``timesteps + 1`` regardless of
        repeated states. The ``encode`` argument forces the states in the
        trajectory to be either encoded or not.  When ``encode is None``,
        whether or not the states of the trajectory are encoded is determined
        by whether or not the initial state (``init``) is provided in encoded
        form.

        Note that when ``timesteps is None``, the length of the resulting
        trajectory should be one greater than the recurrence time of the state.

        If :meth:`landscape` has not been called, this method will implicitly
        call it. Otherwise, it respects any settings provided by such a call.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.trajectory([1,0,0,1,0,1,1,0,1])
            [[1, 0, 0, 1, 0, 1, 1, 0, 1], ... [0, 0, 1, 1, 0, 0, 1, 0, 0]]

            >>> s_pombe.trajectory([1,0,0,1,0,1,1,0,1], encode=True)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> s_pombe.trajectory(361)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76]

            >>> s_pombe.trajectory(361, encode=False)
            [[1, 0, 0, 1, 0, 1, 1, 0, 1], ... [0, 0, 1, 1, 0, 0, 1, 0, 0]]

            >>> s_pombe.trajectory(361, timesteps=5)
            [361, 80, 320, 78, 128, 162]

            >>> s_pombe.trajectory(361, timesteps=10)
            [361, 80, 320, 78, 128, 162, 178, 400, 332, 76, 76]

        :param init: the initial state
        :type init: int or seq
        :param timesteps: the number of time steps to include in the trajectory
        :type timesteps: int or None
        :param encode: whether to encode the states in the trajectory
        :type encode: bool or None
        :return: a list whose elements are subsequent states of the trajectory

        :raises ValueError: if ``init`` an empty array
        :raises ValueError: if ``timesteps`` is less than :math:`1`
        """
        if not self.__landscaped:
            self.landscape()

        decoded = isinstance(init, list) or isinstance(init, np.ndarray)

        if decoded:
            if init == []:
                raise ValueError("initial state cannot be empty")
            elif encode is None:
                encode = False
            init = self.encode(init)
        elif encode is None:
            encode = True

        trans = self.__landscape_data.transitions
        if timesteps is not None:
            if timesteps < 1:
                raise ValueError("number of steps must be positive, non-zero")

            path = [init] * (timesteps + 1)
            for i in range(1, len(path)):
                path[i] = trans[path[i - 1]]
        else:
            path = [init]
            state = trans[init]
            while state not in path:
                path.append(state)
                state = trans[state]

        if not encode:
            decode = self.decode
            path = [decode(state) for state in path]

        return path

    def timeseries(self, timesteps):
        """
        Compute a time series from all states.

        This method computes a 3-dimensional array elements are the states of
        each node in the network. The dimensions of the array are indexed by,
        in order, the node, the initial state and the time step.

        If :meth:`landscape` has not been called, this method will implicitly
        call it. Otherwise, it respects any settings provided by such a call.

        .. rubric:: Basic Usage

        .. doctest:: landscape

            >>> s_pombe.timeseries(5)
            array([[[0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0],
                    ...,
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0]],
            <BLANKLINE>
                   ...
            <BLANKLINE>
                   [[0, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0],
                    ...,
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0]],
            <BLANKLINE>
                   [[0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1],
                    ...,
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0]]])

        :param timesteps: the number of timesteps to evolve the system
        :type timesteps: int
        :return: a 3-D array of node states

        :raises ValueError: if ``timesteps`` is less than :math:`1`
        """
        if not self.__landscaped:
            self.landscape()

        if timesteps < 1:
            raise ValueError("number of steps must be positive, non-zero")

        trans = self.__landscape_data.transitions
        decode = self.decode
        decoded_trans = [decode(state) for state in trans]

        shape = (self.size, self.volume, timesteps + 1)
        series = np.empty(shape, dtype=np.int)

        for index, init in enumerate(self):
            k = index
            series[:, index, 0] = init[:]
            for time in range(1, timesteps + 1):
                series[:, index, time] = decoded_trans[k][:]
                k = trans[k]

        return series

    def expound(self):
        """
        Compute all cached data.

        This function performs the bulk of the calculations that the
        LandscapeMixin is concerned with. Most of the properties in this class
        are computed by this function whenever *any one* of them is requested
        and the results are cached. The advantage of this is that it saves
        computation time; why traverse the state space for every property call
        when you can do it all at once? The downside is that the cached results
        may use a good bit more memory. This is a trade-off that we are willing
        to make for now.

        The properties that are computed by this function include:

        .. autosummary::
           :nosignatures:

           attractors
           attractor_lengths
           basins
           basin_sizes
           basin_entropy
           heights
           recurrence_times
           in_degrees

        """
        if not self.__landscaped:
            self.landscape()

        # Get the state transitions
        trans = self.__landscape_data.transitions
        # Create an array to store whether a given state has visited
        visited = np.zeros(self.volume, dtype=np.bool)
        # Create an array to store which attractor basin each state is in
        basins = np.full(self.volume, -1, dtype=np.int)
        # Create an array to store the in-degree of each state
        in_degrees = np.zeros(self.volume, dtype=np.int)
        # Create an array to store the height of each state
        heights = np.zeros(self.volume, dtype=np.int)
        # Create an array to store the recurrence time of each state
        recurrence_times = np.zeros(self.volume, dtype=np.int)
        # Create a counter to keep track of how many basins have been visited
        basin_number = 0
        # Create a list of basin sizes
        basin_sizes = []
        # Create a list of attractor cycles
        attractors = []
        # Create a list of attractor lengths
        attractor_lengths = []

        # Start at state 0
        initial_state = 0
        # While the initial state is a state of the system
        while initial_state < len(trans):
            # Create a stack to store the state so far visited
            state_stack = []
            # Create a array to store the states in the attractor cycle
            cycle = []
            # Create a flag to signify whether the current state is part of
            # the cycle
            in_cycle = False
            # Set the current state to the initial state
            state = initial_state
            # Store the next state and terminus variables to the next state
            terminus = next_state = trans[state]
            # Set the visited flag of the current state
            visited[state] = True
            # Increment in-degree
            in_degrees[next_state] += 1
            # While the next state hasn't been visited
            while not visited[next_state]:
                # Push the current state onto the stack
                state_stack.append(state)
                # Set the current state to the next state
                state = next_state
                # Update the terminus and next_state variables
                terminus = next_state = trans[state]
                # Update the visited flag for the current state
                visited[state] = True
                # Increment in-degree
                in_degrees[next_state] += 1

            # If the next state hasn't been assigned a basin yet
            if basins[next_state] == -1:
                # Set the current basin to the basin number
                basin = basin_number
                # Increment the basin number
                basin_number += 1
                # Add a new basin size
                basin_sizes.append(0)
                # Add a new attractor length
                attractor_lengths.append(1)
                # Add the current state to the attractor cycle
                cycle.append(state)
                # Set the current state's recurrence time
                recurrence_times[state] = 0
                # We're still in the cycle until the current state is equal to
                # the terminus
                in_cycle = (terminus != state)
            else:
                # Set the current basin to the basin of next_state
                basin = basins[next_state]
                # Set the state's height to one greater than the next state's
                heights[state] = heights[next_state] + 1
                # Set the state's recurrence time to one greater than the next
                # state's
                recurrence_times[state] = recurrence_times[next_state] + 1

            # Set the basin of the current state
            basins[state] = basin
            # Increment the basin size
            basin_sizes[basin] += 1

            # While we still have states on the stack
            while len(state_stack) != 0:
                # Save the current state as the next state
                next_state = state
                # Pop the current state off of the top of the stack
                state = state_stack.pop()
                # Set the basin of the current state
                basins[state] = basin
                # Increment the basin_size
                basin_sizes[basin] += 1
                # If we're still in the cycle
                if in_cycle:
                    # Add the current state to the attractor cycle
                    cycle.append(state)
                    # Increment the current attractor length
                    attractor_lengths[basin] += 1
                    # We're still in the cycle until the current state is
                    # equal to the terminus
                    in_cycle = (terminus != state)
                    # Set the cycle state's recurrence times
                    if not in_cycle:
                        for cycle_state in cycle:
                            rec_time = attractor_lengths[basin] - 1
                            recurrence_times[cycle_state] = rec_time
                else:
                    # Set the state's height to one create than the next
                    # state's
                    heights[state] = heights[next_state] + 1
                    # Set the state's recurrence time to one greater than the
                    # next state's
                    recurrence_times[state] = recurrence_times[next_state] + 1

            # Find the next unvisited initial state
            while initial_state < len(visited) and visited[initial_state]:
                initial_state += 1

            # If the cycle isn't empty, append it to the attractors list
            if len(cycle) != 0:
                attractors.append(np.asarray(cycle, dtype=np.int))

        data = self.__landscape_data

        data.basins = basins
        data.basin_sizes = np.asarray(basin_sizes)
        data.attractors = np.asarray(attractors)
        data.attractor_lengths = np.asarray(attractor_lengths)
        data.in_degrees = in_degrees
        data.heights = heights
        data.recurrence_times = np.asarray(recurrence_times)

        dist = pi.Dist(self.__landscape_data.basin_sizes)
        data.basin_entropy = pi.shannon.entropy(dist, b=2)

        self.__expounded = True

        return self
