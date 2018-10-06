.. automodule:: neet.boolean
    :synopsis: Boolean networks

    .. automodule:: neet.boolean.wtnetwork
        :synopsis: Weight-Threshold networks

        .. autoclass:: neet.boolean.WTNetwork

            .. automethod:: __init__

            .. autoattribute:: size

            .. automethod:: state_space

            .. automethod:: update

            .. automethod:: read

            .. automethod:: neighbors_in

            .. automethod:: neighbors_out

            .. automethod:: neighbors

            .. automethod:: to_networkx_graph

            .. automethod:: draw

            .. automethod:: split_threshold

            .. automethod:: negative_threshold

            .. automethod:: positive_threshold

    .. automodule:: neet.boolean.logicnetwork
        :synopsis: Logic-based networks

        .. autoclass:: neet.boolean.LogicNetwork

            .. automethod:: __init__

            .. autoattribute:: size

            .. automethod:: state_space

            .. automethod:: update

            .. automethod:: reduce_table

            .. automethod:: read_table

            .. automethod:: read_logic

            .. automethod:: neighbors_in

            .. automethod:: neighbors_out

            .. automethod:: neighbors

            .. automethod:: to_networkx_graph

            .. automethod:: draw

    .. automodule:: neet.boolean.conv
        :synopsis: Network Type Conversions

        .. autofunction:: neet.boolean.conv.wt_to_logic 

    .. automodule:: neet.boolean.randomnet
        :synopsis: Network randomization functions

        .. autofunction:: random_logic

    .. automodule:: neet.boolean.examples
        :synopsis: Example Networks

        .. autodata:: s_pombe
          :annotation: = <neet.boolean.WTNetwork object>

        .. autodata:: neet.boolean.examples.s_cerevisiae
          :annotation: = <neet.boolean.WTNetwork object>

        .. autodata:: neet.boolean.examples.c_elegans
          :annotation: = <neet.boolean.WTNetwork object>

        .. autodata:: neet.boolean.examples.p53_no_dmg
          :annotation: = <neet.boolean.WTNetwork object>

        .. autodata:: neet.boolean.examples.p53_dmg
          :annotation: = <neet.boolean.WTNetwork object>

        .. autodata:: neet.boolean.examples.mouse_cortical_7B
          :annotation: = <neet.boolean.LogicNetwork object>

        .. autodata:: neet.boolean.examples.mouse_cortical_7C
          :annotation: = <neet.boolean.LogicNetwork object>

        .. autodata:: neet.boolean.examples.myeloid
          :annotation: = <neet.boolean.LogicNetwork object>
