.. automodule:: neet.synchronous
    :synopsis: Landscape analysis of synchronous networks

    Time Series Generation
    ^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: trajectory

    .. autofunction:: transitions

    .. autofunction:: timeseries

    Synchronous Landscape Analysis Functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autofunction:: transition_graph

    .. autofunction:: attractors

    .. autofunction:: basins

    .. autofunction:: basin_entropy

    Attractor Landscape
    ^^^^^^^^^^^^^^^^^^^

    .. autoclass:: Landscape

        .. automethod:: __init__

        .. autoattribute:: network

        .. autoattribute:: size

        .. autoattribute:: transitions

        .. autoattribute:: attractors

        .. autoattribute:: attractor_lengths

        .. autoattribute:: basins

        .. autoattribute:: basin_sizes

        .. autoattribute:: in_degrees

        .. autoattribute:: heights

        .. autoattribute:: recurrence_times

        .. automethod:: trajectory

        .. automethod:: timeseries

        .. automethod:: basin_entropy
