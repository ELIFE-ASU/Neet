Cellular Automata
=================

API Documentation
-----------------

.. automodule:: neet.automata

Elementary Cellular Automata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.automata.ECA
        :members: __init__, code, boundary, state_space, update, _unsafe_update

Rewired Elementary Cellular Automata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autoclass:: neet.automata.RewiredECA
        :members: __init__, code, boundary, size, wiring, state_space
