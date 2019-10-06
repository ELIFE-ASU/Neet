Network Landscape Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: neet.landscape
    :synopsis: Landscape analysis of synchronous networks

    LandscapeData
    -------------

    .. autoclass:: LandscapeData


    LandscapeMixin
    --------------

    .. autoclass:: LandscapeMixin

        .. automethod:: landscape

        .. automethod:: clear_landscape

        .. autoattribute:: landscape_data

        .. autoattribute:: transitions

        .. autoattribute:: attractors

        .. autoattribute:: attractor_lengths

        .. autoattribute:: basins

        .. autoattribute:: basin_sizes

        .. autoattribute:: basin_entropy

        .. autoattribute:: heights

        .. autoattribute:: recurrence_times

        .. autoattribute:: in_degrees

        .. automethod:: trajectory

        .. automethod:: timeseries

        .. automethod:: landscape_graph

        .. automethod:: draw_landscape_graph

        .. automethod:: expound
