# Copyright 2017 ELIFE. All rights reserved.
# Use of this source code is governed by a MIT
# license that can be found in the LICENSE file.
from .logicnetwork import LogicNetwork
from .wtnetwork import WTNetwork


def wt_to_logic(net):
    """Convert a WTNetwork to a LogicNetwork."""
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
            if (prod > net.thresholds[node]
                    or (prod == net.thresholds[node]
                        and bin_state[indices.index(node)] == '1')):
                conditions.add(bin_state)

        truth_table.append((indices, conditions))

    return LogicNetwork(truth_table, net.names, reduced=True)
