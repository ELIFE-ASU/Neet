Boolean Networks
================

API Documentation
-----------------

.. automodule:: neet.boolean

Weight/Threshold Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.boolean.WTNetwork
        :members: __init__, size, state_space, update, _unsafe_update, read

Threshold Functions
"""""""""""""""""""

        .. automethod:: neet.boolean.WTNetwork.split_threshold

        .. automethod:: neet.boolean.WTNetwork.negative_threshold

        .. automethod:: neet.boolean.WTNetwork.positive_threshold

Logic-based Networks
^^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.boolean.LogicNetwork
        :members: __init__, size, state_space, update, _unsafe_update, read_table

Example Networks
^^^^^^^^^^^^^^^^

    .. automodule:: neet.boolean.examples

Yeast Networks
""""""""""""""

        .. autoattribute:: neet.boolean.examples.s_pombe
        :annotation: = <neet.boolean.WTNetwork object>

        .. autoattribute:: neet.boolean.examples.s_cerevisiae
        :annotation: = <neet.boolean.WTNetwork object>

p53 Networks
""""""""""""

        .. autoattribute:: neet.boolean.examples.p53_no_dmg
        :annotation: = <neet.boolean.WTNetwork object>

        .. autoattribute:: neet.boolean.examples.p53_dmg
        :annotation: = <neet.boolean.WTNetwork object>


Random networks
^^^^^^^^^^^^^^^

    .. automodule:: neet.boolean.rannetwork
