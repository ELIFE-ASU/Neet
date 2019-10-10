"""
.. currentmodule:: neet.boolean.conv

.. testsetup:: conv

    import numpy
    from neet.boolean import LogicNetwork
    from neet.boolean.conv import wt_to_logic
    from neet.boolean.examples import s_pombe
"""
from .logicnetwork import LogicNetwork
from .wtnetwork import WTNetwork


def wt_to_logic(net):
    """
    Convert a :class:`neet.boolean.WTNetwork` to a
    :class:`neet.boolean.LogicNetwork`.

    .. rubric:: Examples

    .. doctest:: conv

        >>> net = wt_to_logic(s_pombe)
        >>> isinstance(net, LogicNetwork)
        True
        >>> numpy.array_equal(net.transitions, s_pombe.transitions)
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
            prod = sum([weights[i] * int(s) for i, s in zip(indices, bin_state)])
            threshold = net.thresholds[node]
            exceeds_threshold = prod > threshold
            at_threshold_and_one = prod == threshold and bin_state[indices.index(node)] == '1'
            if (exceeds_threshold or at_threshold_and_one):
                conditions.add(bin_state)

        truth_table.append((indices, conditions))

    return LogicNetwork(truth_table, reduced=True, names=net.names)
