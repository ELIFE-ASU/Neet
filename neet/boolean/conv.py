"""
.. currentmodule:: neet.boolean.conv

.. testsetup:: conv

    from neet.boolean import LogicNetwork
    from neet.boolean.conv import *
    from neet.boolean.examples import s_pombe
    from neet.synchronous import transitions

Network Type Conversion
=======================

Some algorithms are more efficiently implemented on particular network type.
This module allows the user to convert between network types, though not all
conversions are possible. For example, not every
:class:`neet.boolean.LogicNetwork` can be converted to a
:class:`neet.boolean.WTNetwork`, though the reverse direction is always
possible.
"""
from .logicnetwork import LogicNetwork
from .wtnetwork import WTNetwork


def wt_to_logic(net):
    """
    Convert a :class:`neet.boolean.WTNetwork` to a
    :class:`neet.boolean.LogicNetwork`.

    .. rubric:: Examples

    .. doctest:: conv

        >>> s_pombe_logical = wt_to_logic(s_pombe)
        >>> isinstance(s_pombe_logical, LogicNetwork)
        True
        >>> transitions(s_pombe_logical) == transitions(s_pombe)
        True

    :param net: a network to convert
    :type net: :class:`neet.boolean.WTNetwork`
    :return: an equivalent :class:`neet.boolean.LogicNetwork`
    """
    if not isinstance(net, WTNetwork):
        raise TypeError("'net' must be a WTNetwork")

    truth_table = []
    for node, weights in enumerate(net.weights):
        indices = tuple([i for i, v in enumerate(weights)
                         if v != 0 or i == node])

        conditions = set()
        for dec_state in range(2**len(indices)):
            bin_state = '{0:0{1}b}'.format(dec_state, len(indices))
            prod = sum([weights[i] * int(s)
                        for i, s in zip(indices, bin_state)])
            if (prod > net.thresholds[node] or
                    (prod == net.thresholds[node] and
                        bin_state[indices.index(node)] == '1')):
                conditions.add(bin_state)

        truth_table.append((indices, conditions))

    return LogicNetwork(truth_table, net.names, reduced=True)
