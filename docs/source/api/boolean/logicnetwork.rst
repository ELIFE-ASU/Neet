Logical Networks
----------------

.. automodule:: neet.boolean.logicnetwork
   :synopsis: Logic-based Networks

.. autoclass:: neet.boolean.LogicNetwork

   .. attribute:: table

      The network's truth table.

      A truth table is a list of tuples, one for each node in order. A tuple of
      the form ``(A, {C1, C2, ...})`` at index ``i`` provides the activation
      conditions for the node of index ``i``. ``A`` is a tuple marking the
      indices of the nodes which influence the state of node ``i`` via logic
      relations. ``{C1, C2, ...}`` is a set, each element of which is the
      collection of binary states of these influencing nodes that would
      activate node ``i``, setting it to ``1``. Any other collection of states
      of nodes in ``A`` are assumed to deactivate node ``i``, setting it to
      ``0``.

      ``C1``, ``C2``, etc. are sequences (``tuple`` or ``str``) of binary
      digits, each being the binary state of corresponding node in ``A``.

      .. doctest:: logicnetwork

         >>> from neet.boolean.examples import myeloid
         >>> myeloid.table == [((0, 1, 2, 7), {'1000', '1100', '1010'}),
         ... ((1, 0, 4, 7), {'0010', '1100', '1010', '1110', '0110', '0100', '1000'}),
         ... ((1,), {'1'}),
         ... ((1, 4), {'10'}),
         ... ((1, 3), {'10'}),
         ... ((1, 7), {'10'}),
         ... ((6, 1, 2, 5), {'1011', '1100', '1010', '1110', '1101', '1000', '1001'}),
         ... ((6, 7, 1, 0), {'1000', '1100', '0100'}),
         ... ((7, 10), {'10'}),
         ... ((7, 8, 10), {'110'}),
         ... ((6, 9), {'10'})]
         True

      :type: list of tuples of type (list, set)

   .. automethod:: is_dependent

   .. automethod:: reduce_table

   .. automethod:: read_table

   .. automethod:: read_logic
