.. automodule:: neet.automata
    :synopsis: Automata networks

    .. automodule:: neet.automata.eca
        :synopsis: Elementary cellular automata

        .. autoclass:: ECA

            .. automethod:: __init__

            .. autoattribute:: code

            .. autoattribute:: boundary

            .. automethod:: state_space

            .. automethod:: update

            .. automethod:: neighbors_in

            .. automethod:: neighbors_out

            .. automethod:: neighbors

            .. automethod:: to_networkx_graph

            .. automethod:: draw

    .. automodule:: neet.automata.reca
        :synopsis: Rewired elementary cellular automata

        .. autoclass:: RewiredECA

            .. automethod:: __init__

            .. autoattribute:: code

            .. autoattribute:: boundary

            .. autoattribute:: size

            .. autoattribute:: wiring

            .. automethod:: state_space

            .. automethod:: update

            .. automethod:: neighbors_in

            .. automethod:: neighbors_out

            .. automethod:: neighbors

            .. automethod:: to_networkx_graph

            .. automethod:: draw
