Boolean Networks
================

API Documentation
-----------------

.. automodule:: neet.boolean

Weight/Threshold Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.boolean.WTNetwork

Attributes
""""""""""

        .. autoattribute:: neet.boolean.WTNetwork.size

Initialization
""""""""""""""

        .. automethod:: neet.boolean.WTNetwork.__init__

Methods
"""""""

        .. automethod:: neet.boolean.WTNetwork.state_space

        .. automethod:: neet.boolean.WTNetwork.update

Static Methods
^^^^^^^^^^^^^^

        .. automethod:: neet.boolean.WTNetwork.read

Threshold Functions
^^^^^^^^^^^^^^^^^^^

        .. automethod:: neet.boolean.WTNetwork.split_threshold

        .. automethod:: neet.boolean.WTNetwork.negative_threshold

        .. automethod:: neet.boolean.WTNetwork.positive_threshold
