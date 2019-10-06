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

      .. autoproperty:: landscape_data

      .. autoproperty:: transitions

      .. autoproperty:: attractors

      .. autoproperty:: attractor_lengths

      .. autoproperty:: basins

      .. autoproperty:: basin_sizes

      .. autoproperty:: basin_entropy

      .. autoproperty:: heights

      .. autoproperty:: recurrence_times

      .. autoproperty:: in_degrees

      .. automethod:: trajectory

      .. automethod:: timeseries

      .. automethod:: landscape_graph

      .. automethod:: draw_landscape_graph

      .. automethod:: expound
