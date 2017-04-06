Boolean Networks
================

API Documentation
-----------------

.. automodule:: neet.boolean

Example Networks
^^^^^^^^^^^^^^^^

.. automodule:: neet.boolean.examples

Yeast Networks
""""""""""""""""""

    .. autoattribute:: neet.boolean.examples.s_pombe
        :annotation: = <neet.boolean.WTNetwork object>

    .. autoattribute:: neet.boolean.examples.s_cerevisiae
        :annotation: = <neet.boolean.WTNetwork object>

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

        .. automethod:: neet.boolean.WTNetwork.check_states

Unsafe Methods
""""""""""""""

        .. automethod:: neet.boolean.WTNetwork._unsafe_update

Static Methods
""""""""""""""

        .. automethod:: neet.boolean.WTNetwork.read

Threshold Functions
"""""""""""""""""""

        .. automethod:: neet.boolean.WTNetwork.split_threshold

        .. automethod:: neet.boolean.WTNetwork.negative_threshold

        .. automethod:: neet.boolean.WTNetwork.positive_threshold
