Landscape API
=============

API Documentation
-----------------

.. automodule:: neet.landscape

Interface Tests
^^^^^^^^^^^^^^^

    .. autofunction:: neet.landscape.is_network

    .. autofunction:: neet.landscape.is_fixed_sized

State Generators
^^^^^^^^^^^^^^^^

    .. autoclass:: neet.landscape.StateSpace

        .. automethod:: neet.landscape.StateSpace.__init__

        .. automethod:: neet.landscape.StateSpace.states

        .. automethod:: neet.landscape.StateSpace.encode

        .. automethod:: neet.landscape.StateSpace.decode

Time Series Generation
^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: neet.landscape.trajectory

    .. autofunction:: neet.landscape.transitions

Landscape Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: neet.landscape.transition_graph

    .. autofunction:: neet.landscape.attractors

    .. autofunction:: neet.landscape.basins

