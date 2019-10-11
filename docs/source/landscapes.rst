.. currentmodule:: neet

.. testsetup:: introduction

      from neet.boolean.examples import s_pombe

.. _landscapes:

Attractor Landscapes
====================

The most common use of dynamical network models is the analysis of their attractor landscape. In
many cases, the attractors are associated with some form of functionally important network state,
e.g. a cell type in a gene regulatory network. Neet provides standard landscape analysis methods via
the :class:`LandscapeMixin` from which :class:`Network` derives.

State Transitions
^^^^^^^^^^^^^^^^^

The starting point for all of these analyses are the state `transitions
<api/landscape.html#neet.LandscapeMixin.transitions>`__: where does each state of the network go
upon update?

.. doctest:: introduction

      >>> s_pombe.transitions
      array([  2,   2, 130, 130,   4,   0, 128, 128,   8,   0, 128, 128,  12,
               0, 128, 128, 256, 256, 384, 384, 260, 256, 384, 384, 264, 256,
             ...
             208, 208, 336, 336, 464, 464, 340, 336, 464, 464, 344, 336, 464,
             464, 348, 336, 464, 464])

Each element of the resulting array is the state to which the index transitions, e.g.  :math:`0
\mapsto 2`, :math:`2 \mapsto 130`, etc. The indices and values are, of course, :ref:`encoded
<state-encoding>` states. You can always decode them:

.. doctest:: introduction

      >>> for x, y in enumerate(s_pombe.transitions):
      ...     print(s_pombe.decode(x), '→', s_pombe.decode(y))
      [0, 0, 0, 0, 0, 0, 0, 0, 0] → [0, 1, 0, 0, 0, 0, 0, 0, 0]
      [1, 0, 0, 0, 0, 0, 0, 0, 0] → [0, 1, 0, 0, 0, 0, 0, 0, 0]
      [0, 1, 0, 0, 0, 0, 0, 0, 0] → [0, 1, 0, 0, 0, 0, 0, 1, 0]
      ...
      [1, 1, 1, 1, 1, 1, 1, 1, 1] → [0, 0, 0, 0, 1, 0, 1, 1, 1]

Given state transitions, the next question you might ask is how to compute sequences of state
transtions — a `trajectory <api/landscape.html#neet.LandscapeMixin.trajectory>`__ — by applying the
network update scheme recursively?

.. doctest:: introduction

      >>> s_pombe.trajectory([0, 0, 0, 0, 0, 0, 0, 0, 0], timesteps=2)
      [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0]]
      >>> s_pombe.trajectory([0, 0, 0, 0, 0, 0, 0, 0, 0], timesteps=2, encode=True)
      [0, 2, 130]

Notice that if you request a trajectory with :math:`t` time steps, the resulting trajectory will
have :math:`t+1` elements in it; the first element is the initial state. If you want the trajectory
for *every* state of the network, you can use the `timeseries
<api/landscape.html#neet.LandscapeMixin.timeseries>`__ method.

.. doctest:: introduction

      >>> series = s_pombe.timeseries(2)
      >>> series
      array([[[0, 0, 0],
              [1, 0, 0],
              [0, 0, 0],
              ...,
              [1, 0, 0],
              [0, 0, 0],
              [1, 0, 0]],
      <BLANKLINE>
             ...
      <BLANKLINE>
             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              ...,
              [1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]])
      >>> series.shape
      (9, 512, 3)
      >>> series[:, 0, :].transpose()
      array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0]])

The resulting :math:`3`-D array is indexed by the nodes, state and timestep; in that order. For a
more wholistic description of the state transitions, you can construct a `landscape graph
<api/landscape.html#neet.LandscapeMixin.landscape_graph>`__.

.. doctest:: introduction

      >>> import networkx as nx
      >>> g = s_pombe.landscape_graph()
      >>> len(g)
      512
      >>> nx.shortest_path(g, 0, 130)
      [0, 2, 130]

The landscape graph, much like the network topology, can be drawn if you've installed `pygraphviz
<https://pypi.org/project/pygraphviz/>`_. See `Getting Started
<getting-started.html#dependencies>`__.

Attractors and Basins
^^^^^^^^^^^^^^^^^^^^^

With the state transitions under our belt, we can start computing landscape features such as the
`attractors <api/landscape.html#neet.LandscapeMixin.attractors>`__.


.. doctest:: introduction

      >>> s_pombe.attractors
      array([array([76]), array([4]), array([8]), array([12]),
             array([144, 110, 384]), array([68]), array([72]), array([132]),
             array([136]), array([140]), array([196]), array([200]),
             array([204])], dtype=object)

Each element of the resulting array is an array of states in a fixed-point attractor or limit cycle.
Beyond this, you can determine which of the attractor's `basin
<api/landscape.html#neet.LandscapeMixin.basins>`__ each state is in.

.. doctest:: introduction

      >>> s_pombe.basins
      array([ 0,  0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  0,  3,  0,  0,  0,  0,
              0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  0,  0,  4,  4,  4,  4,
              ...
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0])
      >>> s_pombe.basins[18]
      4

That is, state :math:`18` is in basin :math:`4`, and so is fated to land in the cycle :math:`\{144,
110, 384\}`.

The :class:`LandscapeMixin` provides a whole host of other properties, so check out the `API
Reference <api/landscape.html>`__ for the full list.

Landscape Data
^^^^^^^^^^^^^^

A key feature of the :class:`LandscapeMixin` is that it tries to compute as much as it can, as
efficiently as it can. For example, when the attractors are computed, the basins of all of the
states, the `recurrence time <api/landscape.html#neet.LandscapeMixin.recurrent_times>`__, etc... can
all be computed at the same time. These values are

  1. **Computed lazily, but preemptively** when you first request any of the associated property.
  2. **Cached** in a :class:`LandscapeData` object stored in the :class:`LandscapeMixin`.

This means, that the attractors are computed when you request them. A second request will simply
use the cached values. Similarly, you get a cached value for the basins once you've accessed the
attractors. The following only computes the attractors once, and the basins are computed at that
call:

.. doctest:: introduction

      >>> s_pombe.attractors  # may take a moment
      array([...], dtype=object)
      >>> s_pombe.attractors  # almost instantaneous
      array([...], dtype=object)
      >>> s_pombe.basins  # almost instantaneous; computed on first call to attractors.
      array([...])

The order you access the properties in does not matter, so don't worry about that.

There may be cases when you want to

  1. Compute some landscape features of a network
  2. Modify the network in some way
  3. Compute landscape features on the new network
  4. Compare the results

Because you've modifed the network, you will need to reset the cached landscape data. Since you are
going to be comparing features before and after, you need to extract the data before you do that.
This is where :meth:`LandscapeMixin.landscape`, :meth:`LandscapeMixin.expound` and
:attr:`LandscapeMixin.landscape_data` come into play.

::

      import numpy
      from neet.boolean.examples import s_pombe

      # Compute all of the landscape properties
      s_pombe.expound()
      # Get the data out
      before = s_pombe.landscape_data

      # Modify the network
      s_pombe.thresholds = numpy.zeros(s_pombe.size)
      # Reset the landscape (notice the method chaining...)
      s_pombe.landscape().expound()
      # Get the new data
      after = s_pombe.landscape_data

      # Compare `before` and `after` as you so choose

The result of :attr:`LandscapeMixin.landscape_data` is a :class:`LandscapeData` object which has all
of the landscape features cached (provided they've been computed):

.. doctest:: introduction

      >>> s_pombe.attractors
      array([...], dtype=object)
      >>> s_pombe.landscape_data
      <neet.landscape.LandscapeData object at 0x...>
      >>> s_pombe.landscape_data.attractors
      array([...], dtype=object)
