.. currentmodule:: neet

.. testsetup:: introduction

      import numpy
      from neet.boolean.examples import s_pombe

.. _statespaces:

State Spaces
============

:class:`Network` derives from :class:`StateSpace` which endows it with structural information about
the state space of the network, and provides a number of vital methods.

Attributes
^^^^^^^^^^

First and foremost, :class:`StateSpace` provides (readonly) attributes for assessing gross
properties of the state space, namely :attr:`StateSpace.size`, :attr:`StateSpace.shape` and
:attr:`StateSpace.volume`.

.. doctest:: introduction

      >>> s_pombe.size  # number of dimension (nodes)
      9
      >>> s_pombe.shape  # the number of states by dimension (states per node)
      [2, 2, 2, 2, 2, 2, 2, 2, 2]
      >>> s_pombe.volume  # total number of states of the network
      512

States in the Space
^^^^^^^^^^^^^^^^^^^

As a :class:`StateSpace`, you can determining whether or not an array represents a valid state of
the network. This is accomplished using the ``in`` keyword.

.. doctest:: introduction

      >>> 0 in s_pombe
      False
      >>> [0]*9 in s_pombe
      True
      >>> numpy.zeros(9, dtype=int) in s_pombe
      True
      >>> [2, 0, 0, 0, 0, 0, 0, 0, 0] in s_pombe  # the nodes are binary
      False

Of course, after asking whether a state is valid, the next thing you might want to do is iterate
over the states.

.. doctest:: introduction

      >>> for state in s_pombe:
      ...     print(state)
      [0, 0, 0, 0, 0, 0, 0, 0, 0]
      [1, 0, 0, 0, 0, 0, 0, 0, 0]
      [0, 1, 0, 0, 0, 0, 0, 0, 0]
      [1, 1, 0, 0, 0, 0, 0, 0, 0]
      ...
      [0, 1, 1, 1, 1, 1, 1, 1, 1]
      [1, 1, 1, 1, 1, 1, 1, 1, 1]

Since the networks are iterable, you can treat them like any other kind of sequence.

.. doctest:: introduction

      >>> list(s_pombe)
      [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], ...]
      >>> list(map(lambda s: s[0], s_pombe))
      [0, 1, 0, 1, ...]
      >>> list(filter(lambda s: s[0] ^ s[1] == 1, s_pombe))
      [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], ...]

.. _state-encoding:

State Encoding and Decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For particularly large networks, storing a list of states it's states can use a lot of memory.
What's more, it is often useful to be able to index an array or key a dictionary based by a state of
the network, e.g. when efficiently computing the attractors of the network. A simple solution to
this problem is to encode the state as an integer. :class:`StateSpace` provides this functionality
via the :meth:`StateSpace.encode` and :meth:`StateSpace.decode` methods.

.. rubric:: Encoding States

.. doctest:: introduction

      >>> s_pombe.encode([0, 1, 0, 1, 0, 1, 0, 1, 0])
      170
      >>> s_pombe.encode(numpy.ones(9)) == s_pombe.volume - 1
      True
      >>> s_pombe.encode('apples')
      Traceback (most recent call last):
      ...
      ValueError: state is not in state space

.. rubric:: Decoding States

.. doctest:: introduction

      >>> s_pombe.decode(170)
      [0, 1, 0, 1, 0, 1, 0, 1, 0]
      >>> s_pombe.decode(511)
      [1, 1, 1, 1, 1, 1, 1, 1, 1]
      >>> s_pombe.decode(512)
      [0, 0, 0, 0, 0, 0, 0, 0, 0]
      >>> s_pombe.decode(-1)
      [1, 1, 1, 1, 1, 1, 1, 1, 1]

Notice that decoding states does not raise an error when the state encoding is invalid. Instead, the
codes wrap around so that any integer can be decoded. This was a decision made more for the sake of
performance than anything. Just be mindful of it.

By and large, the :meth:`StateSpace.encode` and :meth:`StateSpace.decode` methods are inverses:

.. doctest:: introduction

      >>> s_pombe.encode(s_pombe.decode(170))
      170
      >>> s_pombe.decode(s_pombe.encode([0, 0, 1, 0, 0, 1, 0, 0, 1]))
      [0, 0, 1, 0, 0, 1, 0, 0, 1]

Encoding Scheme
^^^^^^^^^^^^^^^

There are a number of ways of encoding a sequence of integers as an integer. We've chosen the one we
did so that the encoded value of the state is consistent with the order the states are produced upon
iteration.

.. doctest:: introduction

      >>> states = list(s_pombe)
      >>> states[5] == s_pombe.decode(5)
      True
      >>> numpy.all([i == s_pombe.encode(s) for i, s in enumerate(s_pombe)])
      True
      >>> numpy.all([s_pombe.decode(i) == s for i, s in enumerate(s_pombe)])
      True

This makes implementing the algorithms associated with :ref:`landscape dynamics <landscapes>` and
:ref:`sensitivity analyses <sensitivity>` much simpler and as light on memory as possible.
