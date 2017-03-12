Core API
========

API Documentation
-----------------

.. automodule:: neet

Interface Tests
^^^^^^^^^^^^^^^

    .. autofunction:: neet.is_network

    .. autofunction:: neet.is_fixed_sized

State Generators
^^^^^^^^^^^^^^^^

    .. autoclass:: neet.StateSpace

        .. automethod:: neet.StateSpace.__init__

        .. automethod:: neet.StateSpace.states

        .. automethod:: neet.StateSpace.encode

        .. automethod:: neet.StateSpace.decode


Time Series Generation
^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: neet.trajectory

    .. autofunction:: neet.transitions
