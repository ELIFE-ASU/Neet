Synchronous API
===============

API Documentation
-----------------

.. automodule:: neet.synchronous

Time Series Generation
^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: neet.synchronous.trajectory

    .. autofunction:: neet.synchronous.transitions

    .. autofunction:: neet.synchronous.timeseries

Synchronous Landscape Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: neet.synchronous.transition_graph

    .. autofunction:: neet.synchronous.attractors

    .. autofunction:: neet.synchronous.basins

    .. autofunction:: neet.synchronous.basin_entropy

Attractor Landscape
^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.synchronous.Landscape

        .. automethod:: neet.synchronous.Landscape.__init__

Attributes
""""""""""

        .. autoattribute:: neet.synchronous.Landscape.network

        .. autoattribute:: neet.synchronous.Landscape.size

        .. autoattribute:: neet.synchronous.Landscape.transitions

        .. autoattribute:: neet.synchronous.Landscape.attractors

        .. autoattribute:: neet.synchronous.Landscape.attractor_lengths

        .. autoattribute:: neet.synchronous.Landscape.basins

        .. autoattribute:: neet.synchronous.Landscape.basin_sizes

        .. autoattribute:: neet.synchronous.Landscape.in_degrees

        .. autoattribute:: neet.synchronous.Landscape.heights

        .. autoattribute:: neet.synchronous.Landscape.recurrence_times

Methods
"""""""
        .. automethod:: neet.synchronous.Landscape.trajectory

        .. automethod:: neet.synchronous.Landscape.timeseries

        .. automethod:: neet.synchronous.Landscape.basin_entropy
